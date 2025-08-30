#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iomanip>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <sstream>
#include <queue>
#include <condition_variable>
#include <functional>
#include <fstream>
#include <algorithm>
#include <cmath>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"

ABSL_FLAG(std::string, video, "", "Path to the input video file");
ABSL_FLAG(bool, show_help, false, "Show help message");
ABSL_FLAG(int32_t, threads, 0, "Number of threads to use (0 = auto)");

// Global mutex for thread-safe writing
std::mutex cout_mutex;
std::atomic<int> processed_frames{0};
std::chrono::steady_clock::time_point start_time;
constexpr int CHUNK_SIZE = 100;
constexpr size_t MAX_INPUT_QUEUE_SIZE = 10; // Keep at most 10 chunks in the input queue
constexpr size_t MAX_OUTPUT_QUEUE_SIZE = 100; // Keep at most 100 chunks in the output queue

// Structure to store optical flow analysis results
struct OpticalFlowAnalysis {
    double average_magnitude;
    double movement_consistency;
    double direction_variance;
    int moving_points_count;
    cv::Point2f dominant_direction;
};

// Structure for source chunks from video
struct SourceChunk {
    int chunk_id;
    std::vector<cv::Mat> frames;
    int start_frame;
    int end_frame;
};

// Structure for processed chunks with separated frames
struct ProcessedChunk {
    int chunk_id;
    std::vector<cv::Mat> back_side_frames;
    std::vector<cv::Mat> other_side_frames;
    int start_frame;
    int end_frame;
    
    // Optical flow data for each frame in the chunk
    std::vector<OpticalFlowAnalysis> optical_flow_data;
    
    // For priority queue ordering (lower chunk_id has higher priority)
    bool operator>(const ProcessedChunk& other) const {
        return chunk_id > other.chunk_id;
    }
};

// Thread-safe queue template with backpressure support
template<typename T>
class SynchronizedQueue {
public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
        cv_.notify_one();
    }
    
    bool wait_and_pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || done_; });
        if (queue_.empty()) {
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }
    
    void set_done() {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
        cv_.notify_all();
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cv_;
    bool done_ = false;
};

// Priority queue for ordered writing
template<typename T>
class SynchronizedPriorityQueue {
public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
        cv_.notify_one();
    }
    
    bool wait_and_pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || done_; });
        if (queue_.empty()) {
            return false;
        }
        item = std::move(const_cast<T&>(queue_.top()));
        queue_.pop();
        return true;
    }
    
    bool try_peek(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        item = queue_.top();
        return true;
    }
    
    bool try_pop_if_match(T& item, int expected_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty() || queue_.top().chunk_id != expected_id) {
            return false;
        }
        item = std::move(const_cast<T&>(queue_.top()));
        queue_.pop();
        return true;
    }
    
    void set_done() {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
        cv_.notify_all();
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_;
    std::priority_queue<T, std::vector<T>, std::greater<T>> queue_;
    std::condition_variable cv_;
    bool done_ = false;
};

// Global queues
SynchronizedQueue<SourceChunk> source_chunks;
SynchronizedPriorityQueue<ProcessedChunk> processed_chunks;
std::atomic<int> active_processors{0};

// Forward declarations
std::string format_duration(std::chrono::seconds seconds);
bool is_full_court_view(const cv::Mat& frame);
bool is_full_court_view_temporal(const std::vector<cv::Mat>& frames);
bool is_full_court_view_temporal_with_flow(const std::vector<cv::Mat>& frames, std::vector<OpticalFlowAnalysis>& flow_data);
std::string get_output_path(const std::string& input_path, const std::string& prefix);
void reader_thread(const std::string& video_path, int total_frames);
void processor_thread(SynchronizedQueue<SourceChunk>& input_queue, 
                     SynchronizedPriorityQueue<ProcessedChunk>& output_queue,
                     std::atomic<int>& back_side_count,
                     int total_frames);
void writer_thread(SynchronizedPriorityQueue<ProcessedChunk>& input_queue,
                  cv::VideoWriter& back_writer, cv::VideoWriter& other_writer,
                  std::ofstream& csv_file,
                  std::ofstream& histogram_csv_file, 
                  int total_frames);
void progress_monitor_thread(int total_frames, std::atomic<bool>& processing_complete);
void process_video(const std::string& video_path);

// Function implementations
std::string format_duration(std::chrono::seconds seconds) {
    int hours = seconds.count() / 3600;
    int minutes = (seconds.count() % 3600) / 60;
    int secs = seconds.count() % 60;
    
    std::stringstream ss;
    if (hours > 0) {
        ss << hours << "h ";
    }
    if (minutes > 0 || hours > 0) {
        ss << minutes << "m ";
    }
    ss << secs << "s";
    return ss.str();
}

// Function to calculate optical flow between two frames
OpticalFlowAnalysis calculate_optical_flow(const cv::Mat& prev_frame, const cv::Mat& curr_frame) {
    OpticalFlowAnalysis result = {0.0, 0.0, 0.0, 0, cv::Point2f(0, 0)};
    
    cv::Mat prev_gray, curr_gray;
    cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);
    
    // Detect corner points in the previous frame
    std::vector<cv::Point2f> prev_points, curr_points;
    cv::goodFeaturesToTrack(prev_gray, prev_points, 200, 0.01, 10);
    
    if (prev_points.empty()) {
        return result;
    }
    
    // Calculate optical flow
    std::vector<uchar> status;
    std::vector<float> error;
    cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, curr_points, status, error);
    
    // Analyze flow vectors
    std::vector<cv::Point2f> good_flow_vectors;
    std::vector<double> magnitudes;
    cv::Point2f flow_sum(0, 0);
    
    for (size_t i = 0; i < prev_points.size(); i++) {
        if (status[i] && error[i] < 50) { // Good tracking
            cv::Point2f flow = curr_points[i] - prev_points[i];
            double magnitude = cv::norm(flow);
            
            if (magnitude > 1.0) { // Significant movement threshold
                good_flow_vectors.push_back(flow);
                magnitudes.push_back(magnitude);
                flow_sum += flow;
                result.moving_points_count++;
            }
        }
    }
    
    if (result.moving_points_count == 0) {
        return result;
    }
    
    // Calculate average magnitude
    double total_magnitude = 0;
    for (double mag : magnitudes) {
        total_magnitude += mag;
    }
    result.average_magnitude = total_magnitude / magnitudes.size();
    
    // Calculate dominant direction (average flow vector)
    result.dominant_direction = flow_sum / static_cast<float>(result.moving_points_count);
    
    // Calculate direction consistency (how aligned the flow vectors are)
    cv::Point2f dominant_normalized = result.dominant_direction / cv::norm(result.dominant_direction);
    double alignment_sum = 0;
    for (const cv::Point2f& flow : good_flow_vectors) {
        cv::Point2f flow_normalized = flow / cv::norm(flow);
        double dot_product = dominant_normalized.dot(flow_normalized);
        alignment_sum += std::max(0.0, dot_product); // Only positive alignment
    }
    result.movement_consistency = alignment_sum / good_flow_vectors.size();
    
    // Calculate direction variance
    double variance_sum = 0;
    cv::Point2f mean_direction = result.dominant_direction;
    for (const cv::Point2f& flow : good_flow_vectors) {
        cv::Point2f diff = flow - mean_direction;
        variance_sum += cv::norm(diff) * cv::norm(diff);
    }
    result.direction_variance = std::sqrt(variance_sum / good_flow_vectors.size());
    
    return result;
}

bool is_full_court_view_temporal(const std::vector<cv::Mat>& frames) {
    if (frames.empty()) return false;
    
    std::vector<std::vector<cv::Vec4i>> frame_lines;
    std::vector<int> horizontal_counts;
    std::vector<int> vertical_counts;
    
    // Store optical flow analysis results
    std::vector<OpticalFlowAnalysis> flow_analyses;
    
    // Analyze each frame in the chunk
    for (size_t i = 0; i < frames.size(); i++) {
        const cv::Mat& frame = frames[i];
        
        cv::Mat gray_frame, edges;
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray_frame, gray_frame, cv::Size(5, 5), 0);
        cv::Canny(gray_frame, edges, 50, 150);
        
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 80, 150, 50);
        
        int horizontal_lines = 0;
        int vertical_lines = 0;
        int frame_height = frame.rows;
        int frame_width = frame.cols;
        
        for (const auto& line : lines) {
            int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
            
            double length = std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
            if (length < 100) continue;
            
            double angle = std::atan2(y2 - y1, x2 - x1) * 180.0 / CV_PI;
            angle = std::abs(angle);
            
            bool spans_vertically = std::abs(y2 - y1) > frame_height * 0.2;
            bool spans_horizontally = std::abs(x2 - x1) > frame_width * 0.2;
            
            if (angle < 25 || angle > 155) {
                if (spans_horizontally) horizontal_lines++;
            } else if (angle > 50 && angle < 130) {
                if (spans_vertically) vertical_lines++;
            }
        }
        
        frame_lines.push_back(lines);
        horizontal_counts.push_back(horizontal_lines);
        vertical_counts.push_back(vertical_lines);
        
        // Optical flow analysis (skip first frame)
        if (i > 0) {
            OpticalFlowAnalysis flow = calculate_optical_flow(frames[i-1], frame);
            flow_analyses.push_back(flow);
        } else {
            // Add empty optical flow for first frame
            OpticalFlowAnalysis empty_flow = {0.0, 0.0, 0.0, 0, cv::Point2f(0, 0)};
            flow_analyses.push_back(empty_flow);
        }
    }
    
    // Calculate temporal stability
    double h_mean = 0, v_mean = 0;
    for (int i = 0; i < horizontal_counts.size(); i++) {
        h_mean += horizontal_counts[i];
        v_mean += vertical_counts[i];
    }
    h_mean /= horizontal_counts.size();
    v_mean /= vertical_counts.size();
    
    // Calculate standard deviation (stability measure)
    double h_variance = 0, v_variance = 0;
    for (int i = 0; i < horizontal_counts.size(); i++) {
        h_variance += (horizontal_counts[i] - h_mean) * (horizontal_counts[i] - h_mean);
        v_variance += (vertical_counts[i] - v_mean) * (vertical_counts[i] - v_mean);
    }
    h_variance /= horizontal_counts.size();
    v_variance /= vertical_counts.size();
    
    double h_stability = std::sqrt(h_variance);
    double v_stability = std::sqrt(v_variance);
    
    // Analyze optical flow data
    double avg_camera_movement = 0.0;
    double avg_movement_consistency = 0.0;
    double avg_direction_variance = 0.0;
    int total_moving_points = 0;
    
    if (!flow_analyses.empty()) {
        for (const auto& flow : flow_analyses) {
            avg_camera_movement += flow.average_magnitude;
            avg_movement_consistency += flow.movement_consistency;
            avg_direction_variance += flow.direction_variance;
            total_moving_points += flow.moving_points_count;
        }
        avg_camera_movement /= flow_analyses.size();
        avg_movement_consistency /= flow_analyses.size();
        avg_direction_variance /= flow_analyses.size();
    }
    
    // Camera movement analysis
    bool is_camera_static = avg_camera_movement < 3.0 && avg_movement_consistency > 0.7;
    bool is_camera_moving = avg_camera_movement > 5.0 || avg_direction_variance > 15.0;
    
    // Enhanced back side view detection with optical flow
    bool basic_back_side = (h_mean >= 3) &&           // At least 3 horizontal lines
                          (v_mean >= 8) &&            // Good vertical line count (sidelines)
                          (h_stability < 6.0) &&      // Allow some horizontal variation
                          (v_stability < 10.0) &&     // Allow reasonable vertical variation
                          (h_mean + v_mean >= 15);     // Total line count
    
    // Back side views typically have more static camera angles
    // If camera is very active, it's less likely to be a back side view
    bool optical_flow_favorable = !is_camera_moving || is_camera_static;
    
    // Debug output for optical flow analysis
    static std::mutex flow_debug_mutex;
    std::lock_guard<std::mutex> lock(flow_debug_mutex);
    static int debug_counter = 0;
    if (debug_counter++ % 50 == 0) { // Print every 50th chunk analysis
        std::cout << "\nOptical Flow Analysis:"
                  << " Avg Movement: " << std::fixed << std::setprecision(2) << avg_camera_movement
                  << " Consistency: " << avg_movement_consistency 
                  << " Direction Var: " << avg_direction_variance
                  << " Moving Points: " << total_moving_points
                  << " Static: " << (is_camera_static ? "YES" : "NO")
                  << " Moving: " << (is_camera_moving ? "YES" : "NO") << std::endl;
    }
    
    return basic_back_side && optical_flow_favorable;
}

// Enhanced temporal analysis that also returns optical flow data
bool is_full_court_view_temporal_with_flow(const std::vector<cv::Mat>& frames, std::vector<OpticalFlowAnalysis>& flow_data) {
    if (frames.empty()) return false;
    
    std::vector<std::vector<cv::Vec4i>> frame_lines;
    std::vector<int> horizontal_counts;
    std::vector<int> vertical_counts;
    
    // Store optical flow analysis results
    std::vector<OpticalFlowAnalysis> flow_analyses;
    
    // Analyze each frame in the chunk
    for (size_t i = 0; i < frames.size(); i++) {
        const cv::Mat& frame = frames[i];
        
        cv::Mat gray_frame, edges;
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray_frame, gray_frame, cv::Size(5, 5), 0);
        cv::Canny(gray_frame, edges, 50, 150);
        
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 80, 150, 50);
        
        int horizontal_lines = 0;
        int vertical_lines = 0;
        int frame_height = frame.rows;
        int frame_width = frame.cols;
        
        for (const auto& line : lines) {
            int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
            
            double length = std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
            if (length < 100) continue;
            
            double angle = std::atan2(y2 - y1, x2 - x1) * 180.0 / CV_PI;
            angle = std::abs(angle);
            
            bool spans_vertically = std::abs(y2 - y1) > frame_height * 0.2;
            bool spans_horizontally = std::abs(x2 - x1) > frame_width * 0.2;
            
            if (angle < 25 || angle > 155) {
                if (spans_horizontally) horizontal_lines++;
            } else if (angle > 50 && angle < 130) {
                if (spans_vertically) vertical_lines++;
            }
        }
        
        frame_lines.push_back(lines);
        horizontal_counts.push_back(horizontal_lines);
        vertical_counts.push_back(vertical_lines);
        
        // Optical flow analysis (skip first frame)
        if (i > 0) {
            OpticalFlowAnalysis flow = calculate_optical_flow(frames[i-1], frame);
            flow_analyses.push_back(flow);
        } else {
            // Add empty optical flow for first frame
            OpticalFlowAnalysis empty_flow = {0.0, 0.0, 0.0, 0, cv::Point2f(0, 0)};
            flow_analyses.push_back(empty_flow);
        }
    }
    
    // Copy flow data to output parameter
    flow_data = flow_analyses;
    
    // Calculate temporal stability
    double h_mean = 0, v_mean = 0;
    for (int i = 0; i < horizontal_counts.size(); i++) {
        h_mean += horizontal_counts[i];
        v_mean += vertical_counts[i];
    }
    h_mean /= horizontal_counts.size();
    v_mean /= vertical_counts.size();
    
    // Calculate standard deviation (stability measure)
    double h_variance = 0, v_variance = 0;
    for (int i = 0; i < horizontal_counts.size(); i++) {
        h_variance += (horizontal_counts[i] - h_mean) * (horizontal_counts[i] - h_mean);
        v_variance += (vertical_counts[i] - v_mean) * (vertical_counts[i] - v_mean);
    }
    h_variance /= horizontal_counts.size();
    v_variance /= vertical_counts.size();
    
    double h_stability = std::sqrt(h_variance);
    double v_stability = std::sqrt(v_variance);
    
    // Analyze optical flow data
    double avg_camera_movement = 0.0;
    double avg_movement_consistency = 0.0;
    double avg_direction_variance = 0.0;
    int total_moving_points = 0;
    
    if (!flow_analyses.empty()) {
        for (const auto& flow : flow_analyses) {
            avg_camera_movement += flow.average_magnitude;
            avg_movement_consistency += flow.movement_consistency;
            avg_direction_variance += flow.direction_variance;
            total_moving_points += flow.moving_points_count;
        }
        avg_camera_movement /= flow_analyses.size();
        avg_movement_consistency /= flow_analyses.size();
        avg_direction_variance /= flow_analyses.size();
    }
    
    // Camera movement analysis
    bool is_camera_static = avg_camera_movement < 3.0 && avg_movement_consistency > 0.7;
    bool is_camera_moving = avg_camera_movement > 5.0 || avg_direction_variance > 15.0;
    
    // Enhanced back side view detection with optical flow
    bool basic_back_side = (h_mean >= 3) &&           // At least 3 horizontal lines
                          (v_mean >= 8) &&            // Good vertical line count (sidelines)
                          (h_stability < 6.0) &&      // Allow some horizontal variation
                          (v_stability < 10.0) &&     // Allow reasonable vertical variation
                          (h_mean + v_mean >= 15);     // Total line count
    
    // Back side views typically have more static camera angles
    bool optical_flow_favorable = !is_camera_moving || is_camera_static;
    
    return basic_back_side && optical_flow_favorable;
}

// Fallback single-frame analysis
bool is_full_court_view(const cv::Mat& frame) {
    cv::Mat gray_frame, edges;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray_frame, gray_frame, cv::Size(5, 5), 0);
    cv::Canny(gray_frame, edges, 50, 150);
    
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 80, 150, 50);
    
    int horizontal_lines = 0, vertical_lines = 0;
    int frame_height = frame.rows, frame_width = frame.cols;
    
    for (const auto& line : lines) {
        int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
        double length = std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
        if (length < 100) continue;
        
        double angle = std::atan2(y2 - y1, x2 - x1) * 180.0 / CV_PI;
        angle = std::abs(angle);
        
        bool spans_vertically = std::abs(y2 - y1) > frame_height * 0.2;
        bool spans_horizontally = std::abs(x2 - x1) > frame_width * 0.2;
        
        if (angle < 25 || angle > 155) {
            if (spans_horizontally) horizontal_lines++;
        } else if (angle > 50 && angle < 130) {
            if (spans_vertically) vertical_lines++;
        }
    }
    
    return (horizontal_lines >= 5) && (vertical_lines >= 5) && 
           (horizontal_lines + vertical_lines >= 20);
}

std::string get_output_path(const std::string& input_path, const std::string& prefix) {
    size_t last_slash = input_path.find_last_of("/\\");
    size_t last_dot = input_path.find_last_of(".");
    
    std::string dir = (last_slash != std::string::npos) ? input_path.substr(0, last_slash + 1) : "";
    std::string name = input_path.substr(last_slash + 1, last_dot - last_slash - 1);
    std::string ext = (last_dot != std::string::npos) ? input_path.substr(last_dot) : "";
    
    return dir + prefix + name + ext;
}

// Reader thread that produces source chunks with backpressure control
void reader_thread(const std::string& video_path, int total_frames) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) return;
    
    int chunk_id = 0;
    
    for (int start_frame = 0; start_frame < total_frames; start_frame += CHUNK_SIZE) {
        // Implement backpressure: wait if either queue is too full
        while (source_chunks.size() >= MAX_INPUT_QUEUE_SIZE || 
               processed_chunks.size() >= MAX_OUTPUT_QUEUE_SIZE) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        SourceChunk chunk;
        chunk.chunk_id = chunk_id++;
        chunk.start_frame = start_frame;
        chunk.end_frame = std::min(start_frame + CHUNK_SIZE, total_frames);
        
        cap.set(cv::CAP_PROP_POS_FRAMES, start_frame);
        for (int i = 0; i < CHUNK_SIZE && start_frame + i < total_frames; ++i) {
            cv::Mat frame;
            if (!cap.read(frame)) break;
            chunk.frames.push_back(frame.clone());
        }
        
        source_chunks.push(std::move(chunk));
    }
    
    source_chunks.set_done();
    cap.release();
    
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << "\nReader thread completed. Total chunks: " << chunk_id << std::endl;
}

// Processor thread that consumes source chunks and produces processed chunks
void processor_thread(SynchronizedQueue<SourceChunk>& input_queue, 
                     SynchronizedPriorityQueue<ProcessedChunk>& output_queue,
                     std::atomic<int>& back_side_count,
                     int total_frames) {
    active_processors++;
    SourceChunk source_chunk;
    
    while (input_queue.wait_and_pop(source_chunk)) {
        ProcessedChunk result;
        result.chunk_id = source_chunk.chunk_id;
        result.start_frame = source_chunk.start_frame;
        result.end_frame = source_chunk.end_frame;
        
        // Calculate optical flow for frame-by-frame classification
        std::vector<OpticalFlowAnalysis> flow_data;
        
        // Compute optical flow between consecutive frames
        for (size_t i = 1; i < source_chunk.frames.size(); i++) {
            OpticalFlowAnalysis flow_analysis = calculate_optical_flow(source_chunk.frames[i-1], source_chunk.frames[i]);
            flow_data.push_back(flow_analysis);
        }
        
        // Store optical flow data for CSV output
        result.optical_flow_data = flow_data;
        
        int local_back_side = 0;
        for (size_t i = 0; i < source_chunk.frames.size(); i++) {
            bool is_back_side_frame = false;
            
            if (i == 0) {
                // For first frame, use next frame's optical flow if available
                if (flow_data.size() > 0) {
                    is_back_side_frame = (flow_data[0].moving_points_count >= 0 && flow_data[0].moving_points_count <= 60);
                } else {
                    is_back_side_frame = true; // Default to back side if no flow data
                }
            } else {
                // For other frames, use the optical flow from previous frame to current frame
                size_t flow_index = i - 1;
                if (flow_index < flow_data.size()) {
                    is_back_side_frame = (flow_data[flow_index].moving_points_count >= 0 && flow_data[flow_index].moving_points_count <= 60);
                } else {
                    is_back_side_frame = true; // Default to back side if no flow data
                }
            }
            
            if (is_back_side_frame) {
                result.back_side_frames.push_back(source_chunk.frames[i].clone());
                local_back_side++;
            } else {
                result.other_side_frames.push_back(source_chunk.frames[i].clone());
            }
            
            processed_frames++;
        }
        
        back_side_count += local_back_side;
        output_queue.push(std::move(result));
    }
    
    if (--active_processors == 0) {
        output_queue.set_done();
    }
}

// Writer thread that writes processed chunks in strict sequential order
void writer_thread(SynchronizedPriorityQueue<ProcessedChunk>& input_queue,
                  cv::VideoWriter& back_writer, cv::VideoWriter& other_writer,
                  std::ofstream& csv_file,
                  std::ofstream& histogram_csv_file, 
                  int total_frames) {
    int expected_chunk_id = 0;
    int total_chunks = (total_frames + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    // Collect all frame magnitudes for final histogram calculation
    std::vector<double> all_frame_magnitudes;
    all_frame_magnitudes.reserve(total_frames);
    
    while (expected_chunk_id < total_chunks) {
        ProcessedChunk chunk;
        
        // Try to get the chunk we're expecting
        if (input_queue.try_pop_if_match(chunk, expected_chunk_id)) {
            // Write the chunk since it's the one we expected
            for (const auto& frame : chunk.back_side_frames) {
                back_writer.write(frame);
            }
            for (const auto& frame : chunk.other_side_frames) {
                other_writer.write(frame);
            }
            
            // Write optical flow data to CSV and collect magnitudes
            for (size_t i = 0; i < chunk.optical_flow_data.size(); i++) {
                const auto& flow = chunk.optical_flow_data[i];
                int frame_number = chunk.start_frame + i;
                
                csv_file << frame_number << ","
                        << flow.average_magnitude << ","
                        << flow.movement_consistency << ","
                        << flow.direction_variance << ","
                        << flow.moving_points_count << ","
                        << flow.dominant_direction.x << ","
                        << flow.dominant_direction.y << "\n";
                
                // Collect magnitude for histogram (skip frames with no motion)
                if (flow.moving_points_count > 0 && flow.average_magnitude >= 1.0) {
                    all_frame_magnitudes.push_back(flow.average_magnitude);
                }
            }
            
            expected_chunk_id++;
            
            // Log progress for last chunk
            if (expected_chunk_id == total_chunks) {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "\nWriter: All chunks written successfully. Last chunk: " 
                          << (expected_chunk_id - 1) << std::endl;
            }
        } else {
            // The chunk we need isn't ready yet, sleep and try again
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    // Now calculate and write the histogram
    if (all_frame_magnitudes.empty()) {
        histogram_csv_file << "bin_start,bin_end,frame_count\n";
        histogram_csv_file << "# No frames with motion detected\n";
        return;
    }
    
    const int num_bins = 50;
    
    // First pass: find actual min/max values
    double max_magnitude = *std::max_element(all_frame_magnitudes.begin(), all_frame_magnitudes.end());
    double min_magnitude = *std::min_element(all_frame_magnitudes.begin(), all_frame_magnitudes.end());
    
    // Use logarithmic binning for better distribution
    // Add small epsilon to avoid log(0)
    double log_min = std::log(min_magnitude + 0.001);
    double log_max = std::log(max_magnitude + 0.001);
    double log_bin_width = (log_max - log_min) / num_bins;
    
    std::vector<int> histogram_bins(num_bins, 0);
    std::vector<double> bin_edges(num_bins + 1);
    
    // Calculate exponential bin edges
    for (int i = 0; i <= num_bins; i++) {
        bin_edges[i] = std::exp(log_min + i * log_bin_width) - 0.001;
    }
    
    // Second pass: calculate histogram with exponential bins
    for (double magnitude : all_frame_magnitudes) {
        // Find which bin this magnitude belongs to using binary search-like approach
        int bin_index = -1;
        for (int i = 0; i < num_bins; i++) {
            if (magnitude >= bin_edges[i] && magnitude < bin_edges[i + 1]) {
                bin_index = i;
                break;
            }
        }
        // Handle edge case for maximum value
        if (bin_index == -1 && magnitude >= bin_edges[num_bins - 1]) {
            bin_index = num_bins - 1;
        }
        
        if (bin_index >= 0) {
            histogram_bins[bin_index]++;
        }
    }
    
    // Write histogram header
    histogram_csv_file << "bin_start,bin_end,frame_count\n";
    
    // Write histogram data
    for (int i = 0; i < num_bins; i++) {
        histogram_csv_file << std::fixed << std::setprecision(4) 
                          << bin_edges[i] << "," << bin_edges[i + 1] << "," << histogram_bins[i] << "\n";
    }
    
    // Write summary statistics as comments
    histogram_csv_file << "# Total frames with motion: " << all_frame_magnitudes.size() << "\n";
    histogram_csv_file << "# Max magnitude: " << std::fixed << std::setprecision(4) << max_magnitude << "\n";
    histogram_csv_file << "# Min magnitude: " << std::fixed << std::setprecision(4) << min_magnitude << "\n";
    histogram_csv_file << "# Binning: Exponential/logarithmic for better low-value resolution\n";
}

// Progress monitoring thread
void progress_monitor_thread(int total_frames, std::atomic<bool>& processing_complete) {
    while (!processing_complete) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        if (processed_frames > 0) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            float progress = (float)processed_frames / total_frames * 100;
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
            
            float frames_per_second = (float)processed_frames / elapsed.count();
            int remaining_frames = total_frames - processed_frames;
            auto eta = std::chrono::seconds(static_cast<int>(remaining_frames / frames_per_second));
            
            std::cout << "\rProgress: " << std::fixed << std::setprecision(1) 
                     << progress << "% (" << processed_frames << "/" 
                     << total_frames << " frames) "
                     << "[" << frames_per_second << " fps] "
                     << "Input Q: " << source_chunks.size() 
                     << " Output Q: " << processed_chunks.size() << " chunks "
                     << "ETA: " << format_duration(eta) << "     " << std::flush;
        }
    }
}

void process_video(const std::string& video_path) {
    std::cout << "Opening video file: " << video_path << std::endl;
    
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return;
    }

    // Get video properties
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    int expected_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

    std::cout << "Video properties:" << std::endl
              << "  Resolution: " << frame_width << "x" << frame_height << std::endl
              << "  FPS: " << fps << std::endl
              << "  Total frames: " << expected_frames << std::endl;

    // Create output paths
    std::string back_side_path = get_output_path(video_path, "back_side_");
    std::string other_side_path = get_output_path(video_path, "other_side_");
    std::string csv_path = get_output_path(video_path, "optical_flow_");
    std::string histogram_csv_path = get_output_path(video_path, "optical_flow_histogram_");
    
    // Replace video extension with .csv for the CSV files
    size_t dot_pos = csv_path.find_last_of(".");
    if (dot_pos != std::string::npos) {
        csv_path = csv_path.substr(0, dot_pos) + ".csv";
        histogram_csv_path = histogram_csv_path.substr(0, histogram_csv_path.find_last_of(".")) + ".csv";
    } else {
        csv_path += ".csv";
        histogram_csv_path += ".csv";
    }
    
    std::cout << "Creating output files:" << std::endl
              << "  Back side views: " << back_side_path << std::endl
              << "  Other views: " << other_side_path << std::endl
              << "  Optical flow CSV: " << csv_path << std::endl
              << "  Optical flow histogram CSV: " << histogram_csv_path << std::endl;

    // Create video writers
    cv::VideoWriter back_side_writer(
        back_side_path,
        fourcc, fps, cv::Size(frame_width, frame_height)
    );
    cv::VideoWriter other_side_writer(
        other_side_path,
        fourcc, fps, cv::Size(frame_width, frame_height)
    );

    // Create CSV file for optical flow data
    std::ofstream csv_file(csv_path);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Could not create CSV file: " << csv_path << std::endl;
        return;
    }
    
    // Create CSV file for histogram data
    std::ofstream histogram_csv_file(histogram_csv_path);
    if (!histogram_csv_file.is_open()) {
        std::cerr << "Error: Could not create histogram CSV file: " << histogram_csv_path << std::endl;
        return;
    }
    
    // Write CSV headers
    csv_file << "frame_number,average_magnitude,movement_consistency,direction_variance,moving_points_count,dominant_direction_x,dominant_direction_y\n";
    
    // Histogram CSV header will be written by writer thread

    if (!back_side_writer.isOpened() || !other_side_writer.isOpened()) {
        std::cerr << "Error: Could not create output video files." << std::endl;
        return;
    }

    // Reset global states
    processed_frames = 0;
    active_processors = 0;
    start_time = std::chrono::steady_clock::now();

    // Determine number of processor threads
    int num_threads = absl::GetFlag(FLAGS_threads);
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency() - 2;
    }
    std::cout << "Using " << num_threads << " threads..." << std::endl;

    // Create threads
    std::vector<std::thread> processor_threads;
    std::atomic<int> back_side_frames{0};
    std::atomic<bool> processing_complete{false};

    // Start progress monitoring thread
    std::thread progress_monitor(progress_monitor_thread, expected_frames, std::ref(processing_complete));

    // Start reader thread
    std::thread reader(reader_thread, std::ref(video_path), expected_frames);

    // Start processor threads
    for (int i = 0; i < num_threads; ++i) {
        processor_threads.emplace_back(processor_thread, 
                                     std::ref(source_chunks),
                                     std::ref(processed_chunks),
                                     std::ref(back_side_frames),
                                     expected_frames);
    }

    // Start writer thread
    std::thread writer(writer_thread, 
                      std::ref(processed_chunks),
                      std::ref(back_side_writer), 
                      std::ref(other_side_writer),
                      std::ref(csv_file),
                      std::ref(histogram_csv_file),
                      expected_frames);

    // Wait for all threads to complete
    reader.join();
    for (auto& thread : processor_threads) {
        thread.join();
    }
    writer.join();
    
    // Stop progress monitoring
    processing_complete = true;
    progress_monitor.join();

    auto end_time = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    float avg_fps = (float)processed_frames / total_time.count();
    
    std::cout << "\n\nProcessing complete!" << std::endl;
    std::cout << "Total time: " << format_duration(total_time) << std::endl;
    std::cout << "Average speed: " << std::fixed << std::setprecision(1) << avg_fps << " fps" << std::endl;

    std::cout << "Summary:" << std::endl;
    std::cout << "  Total frames processed: " << processed_frames << std::endl;
    std::cout << "  Back side views: " << back_side_frames 
              << " (" << std::fixed << std::setprecision(1) 
              << (float)back_side_frames/processed_frames*100 << "%)" << std::endl;
    std::cout << "  Other views: " << (processed_frames - back_side_frames)
              << " (" << std::fixed << std::setprecision(1) 
              << (float)(processed_frames - back_side_frames)/processed_frames*100 << "%)" << std::endl;
    std::cout << "\nOutput files created successfully." << std::endl;

    // Release resources
    cap.release();
    back_side_writer.release();
    other_side_writer.release();
    csv_file.close();
    histogram_csv_file.close();
}

int main(int argc, char* argv[]) {
    // Set up usage message
    absl::SetProgramUsageMessage(
        "Detect full court views in a badminton video.\n"
        "Usage: detect_angle --video=<path_to_video>\n"
        "Example: detect_angle --video=match.mp4"
    );

    // Parse command line flags
    absl::ParseCommandLine(argc, argv);

    // Check if help was requested
    if (absl::GetFlag(FLAGS_show_help)) {
        std::cout << absl::ProgramUsageMessage() << std::endl;
        return 0;
    }

    // Get the video path
    std::string video_path = absl::GetFlag(FLAGS_video);
    
    // If no video path provided, try to get it from positional arguments
    if (video_path.empty() && argc > 1) {
        video_path = argv[1];
    }

    if (video_path.empty()) {
        std::cerr << "Error: No video file specified\n"
                  << absl::ProgramUsageMessage() << std::endl;
        return 1;
    }

    process_video(video_path);
    return 0;
}

#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <cmath>
#include <string>
#include <sstream>
#include <algorithm>

// Enhanced scene classification based on feature vectors
class SceneClassifier {
private:
    // Predefined feature patterns for different scene types
    std::map<std::string, std::vector<float>> scene_patterns;
    
public:
    SceneClassifier() {
        // Initialize with some basic patterns (you can enhance these with real data)
        // These are example patterns - in practice, you'd train these on labeled data
        
        // Court view pattern (high values in certain regions)
        scene_patterns["court_view"] = std::vector<float>(4096, 0.1f);
        for (int i = 0; i < 1000; ++i) scene_patterns["court_view"][i] = 0.2f;
        for (int i = 2000; i < 3000; ++i) scene_patterns["court_view"][i] = 0.15f;
        
        // Back side view pattern
        scene_patterns["back_side"] = std::vector<float>(4096, 0.05f);
        for (int i = 1000; i < 2000; ++i) scene_patterns["back_side"][i] = 0.25f;
        for (int i = 3500; i < 4096; ++i) scene_patterns["back_side"][i] = 0.1f;
        
        // Other angle pattern
        scene_patterns["other_angle"] = std::vector<float>(4096, 0.08f);
        for (int i = 500; i < 1500; ++i) scene_patterns["other_angle"][i] = 0.18f;
        for (int i = 2500; i < 3500; ++i) scene_patterns["other_angle"][i] = 0.12f;
        
        // Close-up pattern
        scene_patterns["close_up"] = std::vector<float>(4096, 0.12f);
        for (int i = 0; i < 500; ++i) scene_patterns["close_up"][i] = 0.3f;
        for (int i = 3000; i < 4096; ++i) scene_patterns["close_up"][i] = 0.2f;
    }
    
    std::string classify_scene(const std::vector<float>& features) {
        std::string best_match = "unknown";
        double best_similarity = -1.0;
        
        for (const auto& pattern : scene_patterns) {
            double similarity = cosine_similarity(features, pattern.second);
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_match = pattern.first;
            }
        }
        
        // Additional heuristics based on feature statistics
        double mean = 0.0, variance = 0.0;
        for (float f : features) mean += f;
        mean /= features.size();
        
        for (float f : features) variance += (f - mean) * (f - mean);
        variance /= features.size();
        
        // Refine classification based on statistics
        if (best_similarity < 0.3) {
            if (variance > 0.15) return "complex_scene";
            if (mean > 0.1) return "bright_scene";
            return "simple_scene";
        }
        
        return best_match;
    }
    
    double cosine_similarity(const std::vector<float>& vec1, const std::vector<float>& vec2) {
        if (vec1.size() != vec2.size()) return 0.0;
        
        double dot_product = 0.0, norm1 = 0.0, norm2 = 0.0;
        for (size_t i = 0; i < vec1.size(); ++i) {
            dot_product += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }
        
        norm1 = std::sqrt(norm1);
        norm2 = std::sqrt(norm2);
        
        if (norm1 == 0.0 || norm2 == 0.0) return 0.0;
        return dot_product / (norm1 * norm2);
    }
    
    // Analyze scene characteristics
    std::map<std::string, double> analyze_scene_characteristics(const std::vector<float>& features) {
        std::map<std::string, double> characteristics;
        
        // Basic statistics
        double mean = 0.0, variance = 0.0, min_val = 1e6, max_val = -1e6;
        for (float f : features) {
            mean += f;
            min_val = std::min(min_val, (double)f);
            max_val = std::max(max_val, (double)f);
        }
        mean /= features.size();
        
        for (float f : features) {
            variance += (f - mean) * (f - mean);
        }
        variance /= features.size();
        
        characteristics["mean"] = mean;
        characteristics["variance"] = variance;
        characteristics["min"] = min_val;
        characteristics["max"] = max_val;
        characteristics["range"] = max_val - min_val;
        
        // Analyze feature distribution
        int low_features = 0, mid_features = 0, high_features = 0;
        for (float f : features) {
            if (f < mean - std::sqrt(variance)) low_features++;
            else if (f > mean + std::sqrt(variance)) high_features++;
            else mid_features++;
        }
        
        characteristics["low_features_ratio"] = (double)low_features / features.size();
        characteristics["mid_features_ratio"] = (double)mid_features / features.size();
        characteristics["high_features_ratio"] = (double)high_features / features.size();
        
        return characteristics;
    }
    
    // Load feature vectors from CSV file
    std::vector<std::vector<float>> load_features_from_csv(const std::string& filename) {
        std::vector<std::vector<float>> features;
        std::ifstream file(filename);
        
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return features;
        }
        
        std::string line;
        bool first_line = true;
        
        while (std::getline(file, line)) {
            if (first_line) {
                first_line = false;
                continue; // Skip header
            }
            
            std::vector<float> feature_vector;
            std::stringstream ss(line);
            std::string token;
            
            // Skip frame and timestamp columns
            std::getline(ss, token, ',');
            std::getline(ss, token, ',');
            
            // Parse feature values
            while (std::getline(ss, token, ',')) {
                try {
                    feature_vector.push_back(std::stof(token));
                } catch (...) {
                    std::cerr << "Warning: Could not parse feature value: " << token << std::endl;
                }
            }
            
            if (feature_vector.size() == 4096) {
                features.push_back(feature_vector);
            }
        }
        
        return features;
    }
};

int main() {
    try {
        std::cout << "DINOv3 Scene Classifier Demo" << std::endl;
        std::cout << "============================" << std::endl;
        
        SceneClassifier classifier;
        
        // Test with some sample feature vectors
        std::vector<std::vector<float>> test_features = {
            std::vector<float>(4096, 0.1f),  // Simple scene
            std::vector<float>(4096, 0.2f),  // Bright scene
            std::vector<float>(4096, 0.05f), // Dark scene
        };
        
        // Modify test features to simulate different scenes
        for (int i = 0; i < 1000; ++i) test_features[0][i] = 0.2f;  // Court view pattern
        for (int i = 1000; i < 2000; ++i) test_features[1][i] = 0.25f; // Back side pattern
        for (int i = 0; i < 500; ++i) test_features[2][i] = 0.3f;   // Close-up pattern
        
        std::vector<std::string> scene_names = {"Test Scene 1", "Test Scene 2", "Test Scene 3"};
        
        for (size_t i = 0; i < test_features.size(); ++i) {
            std::cout << "\n" << scene_names[i] << ":" << std::endl;
            
            std::string scene_type = classifier.classify_scene(test_features[i]);
            std::cout << "  Classified as: " << scene_type << std::endl;
            
            auto characteristics = classifier.analyze_scene_characteristics(test_features[i]);
            std::cout << "  Characteristics:" << std::endl;
            std::cout << "    Mean: " << characteristics["mean"] << std::endl;
            std::cout << "    Variance: " << characteristics["variance"] << std::endl;
            std::cout << "    Range: " << characteristics["range"] << std::endl;
            std::cout << "    Low features ratio: " << characteristics["low_features_ratio"] << std::endl;
            std::cout << "    High features ratio: " << characteristics["high_features_ratio"] << std::endl;
        }
        
        // Try to load real features if available
        std::cout << "\nTrying to load real feature vectors..." << std::endl;
        auto real_features = classifier.load_features_from_csv("frame_features.csv");
        
        if (!real_features.empty()) {
            std::cout << "Loaded " << real_features.size() << " feature vectors from frame_features.csv" << std::endl;
            
            for (size_t i = 0; i < std::min(real_features.size(), size_t(3)); ++i) {
                std::cout << "\nReal Frame " << i << ":" << std::endl;
                
                std::string scene_type = classifier.classify_scene(real_features[i]);
                std::cout << "  Classified as: " << scene_type << std::endl;
                
                auto characteristics = classifier.analyze_scene_characteristics(real_features[i]);
                std::cout << "  Mean: " << characteristics["mean"] << std::endl;
                std::cout << "  Variance: " << characteristics["variance"] << std::endl;
            }
        } else {
            std::cout << "No frame_features.csv found. Run video_scene_detector first to generate feature vectors." << std::endl;
        }
        
        std::cout << "\nScene classifier demo completed!" << std::endl;
        std::cout << "\nTo use with real video data:" << std::endl;
        std::cout << "1. Run: ./build/video_scene_detector your_video.mp4" << std::endl;
        std::cout << "2. This will generate frame_features.csv" << std::endl;
        std::cout << "3. Run this classifier again to analyze the real features" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}

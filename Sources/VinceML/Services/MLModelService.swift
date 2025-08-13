//
//  MLModelService.swift
//  VinceML
//
//  Created by Vince Carlo Santos on 8/13/25.
//

import Foundation
import CoreML
import CreateML
import Vision
// Platform-agnostic image import
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#else
#error("Unsupported platform: VinceML requires UIKit or AppKit for image handling.")
#endif

// MARK: - Protocol Definition

/// Service protocol for machine learning model operations
///
/// Provides core functionality for training image classification models
/// and performing inference on images using trained models.
public protocol MLModelServiceProtocol {
    /// Trains a new image classification model from provided training data
    /// - Parameters:
    ///   - trainingDataURL: URL to directory containing training images organized by label folders
    ///   - saveURL: URL where the trained model should be saved
    /// - Returns: The trained MLModel instance
    /// - Throws: MLServiceError if training fails or data is invalid
    func trainAndSaveModel(from trainingDataURL: URL, to saveURL: URL) async throws -> MLModel
    
    /// Classifies an image using a trained model
    /// - Parameters:
    ///   - image: VinceMLImage to classify
    ///   - model: Trained MLModel to use for classification
    /// - Returns: Array of classification results with confidence scores
    func classifyImage(_ image: VinceMLImage, using model: MLModel) async -> [String]
}

// MARK: - Service Implementation

/// MLModelService handles machine learning model training and inference operations
///
/// This service provides a high-level interface for:
/// - Training image classification models using CreateML
/// - Performing image classification using trained models
/// - Validating training data structure and quality
///
/// **Training Requirements:**
/// - At least 2 categories (label folders)
/// - Minimum 5 images per category
/// - Supported formats: JPG, JPEG, PNG, HEIC
///
/// **Classification Process:**
/// 1. Converts UIImage to CGImage for Vision framework
/// 2. Uses VNCoreMLModel for efficient inference
/// 3. Returns top 3 predictions with confidence scores
///
/// **Example Usage:**
/// ```swift
/// import VinceML
/// 
/// let service = MLModelService()
/// 
/// // Training
/// let model = try await service.trainAndSaveModel(
///     from: trainingDataURL,
///     to: modelSaveURL
/// )
/// 
/// // Classification
/// let results = await service.classifyImage(image, using: model)
/// ```
public class MLModelService: MLModelServiceProtocol, ObservableObject {
    
    private let fileManager: FileManager
    
    /// Initialize the MLModelService
    /// - Parameter fileManager: FileManager instance for file operations (default: .default)
    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }
    
    /// Trains and saves an image classification model
    /// 
    /// Creates a new MLImageClassifier using the provided training data
    /// and saves it to the specified location. Training data must be organized
    /// in a specific folder structure with label folders containing images.
    ///
    /// - Parameters:
    ///   - trainingDataURL: URL to folder containing label subfolders with training images
    ///   - saveURL: URL where the trained .mlmodel file should be saved
    /// - Returns: The trained MLModel instance ready for inference
    /// - Throws: MLServiceError for validation failures or training errors
    public func trainAndSaveModel(from trainingDataURL: URL, to saveURL: URL) async throws -> MLModel {
        // Validate training data structure and quality before training
        try validateTrainingData(at: trainingDataURL)
        
        // Configure training with data augmentation for better model robustness
        let trainingJob = try MLImageClassifier(
            trainingData: .labeledDirectories(at: trainingDataURL),
            parameters: .init(augmentation: [.blur, .crop, .exposure, .flip, .noise, .rotation])
        )
        
        // Save the trained classifier to specified location
        try trainingJob.write(to: saveURL)
        
        return trainingJob.model
    }
    
    /// Classifies an image using a trained CoreML model
    ///
    /// Performs image classification using Vision framework with the provided model.
    /// The image is processed through VNCoreMLModel for optimal performance and
    /// returns the top 3 classification results with confidence percentages.
    ///
    /// **Process Overview:**
    /// 1. Validates and converts UIImage to CGImage
    /// 2. Creates VNCoreMLModel wrapper for the MLModel
    /// 3. Performs inference using VNImageRequestHandler
    /// 4. Extracts and formats top classification results
    ///
    /// - Parameters:
    ///   - image: UIImage to classify
    ///   - model: Trained MLModel for classification
    /// - Returns: Array of formatted classification results (max 3), each containing
    ///           label and confidence percentage (e.g., "Cat: 95.67%")
    public func classifyImage(_ image: VinceMLImage, using model: MLModel) async -> [String] {
#if canImport(UIKit)
        guard let cgImage = image.cgImage else {
            return ["Invalid image format - unable to process"]
        }
#elseif canImport(AppKit)
        var imageRect = CGRect(origin: .zero, size: image.size)
        guard let cgImage = image.cgImage(forProposedRect: &imageRect, context: nil, hints: nil) else {
            return ["Invalid image format - unable to process"]
        }
#else
        return ["Unsupported platform"]
#endif
        do {
            let vnModel = try VNCoreMLModel(for: model)
            let request = VNCoreMLRequest(model: vnModel)
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            try handler.perform([request])
            guard let results = request.results as? [VNClassificationObservation] else {
                return ["No classification results available"]
            }
            let topResults = Array(results.prefix(3)).map {
                "\($0.identifier): \(String(format: "%.2f", $0.confidence * 100))%"
            }
            return topResults.isEmpty ? ["No predictions available"] : topResults
        } catch {
            return ["Classification failed: \(error.localizedDescription)"]
        }
    }
    
    // MARK: - Private Methods
    
    /// Validates training data structure and quality requirements
    ///
    /// Ensures the training data meets minimum requirements for successful
    /// model training including directory structure, category count, and
    /// sufficient images per category.
    ///
    /// **Validation Criteria:**
    /// - Training directory exists and is accessible
    /// - At least 2 category subdirectories present
    /// - Each category contains minimum 5 training images
    /// - Images are in supported formats (JPG, JPEG, PNG, HEIC)
    ///
    /// - Parameter url: URL to the training data directory
    /// - Throws: MLServiceError for any validation failure
    private func validateTrainingData(at url: URL) throws {
        // Verify training data directory exists
        guard fileManager.fileExists(atPath: url.path) else {
            throw MLServiceError.trainingDataNotFound
        }
        
        // Get all subdirectories (categories) in training data folder
        let contents = try fileManager.contentsOfDirectory(at: url, includingPropertiesForKeys: [.isDirectoryKey])
        let categoryDirectories = contents.filter { url in
            let isDirectory = try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory
            return isDirectory == true
        }
        
        // Require at least 2 categories for classification training
        guard categoryDirectories.count >= 2 else {
            throw MLServiceError.insufficientCategories
        }
        
        // Validate each category has sufficient training images
        for categoryDir in categoryDirectories {
            let images = try fileManager.contentsOfDirectory(at: categoryDir, includingPropertiesForKeys: nil)
                .filter { url in
                    let pathExtension = url.pathExtension.lowercased()
                    return ["jpg", "jpeg", "png", "heic"].contains(pathExtension)
                }
            
            // Require minimum 5 images per category for effective training
            guard images.count >= 5 else {
                throw MLServiceError.insufficientImagesInCategory(categoryDir.lastPathComponent)
            }
        }
    }
}

// MARK: - Service Errors
public enum MLServiceError: LocalizedError {
    case invalidImage
    case noResults
    case trainingDataNotFound
    case insufficientCategories
    case insufficientImagesInCategory(String)
    case modelSavingNotSupported
    case downloadFailed
    case compilationFailed
    
    public var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "Invalid image format"
        case .noResults:
            return "No classification results"
        case .trainingDataNotFound:
            return "Training data directory not found"
        case .insufficientCategories:
            return "Need at least 2 categories for training"
        case .insufficientImagesInCategory(let category):
            return "Category '\(category)' needs at least 5 images"
        case .modelSavingNotSupported:
            return "Direct MLModel saving not supported. Use trained classifier's write method instead."
        case .downloadFailed:
            return "Failed to download model from remote URL"
        case .compilationFailed:
            return "Failed to compile downloaded model"
        }
    }
}

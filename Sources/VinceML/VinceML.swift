// Platform-agnostic image typealias for all Apple platforms
#if canImport(UIKit)
import UIKit
public typealias VinceMLImage = UIImage
#elseif canImport(AppKit)
import AppKit
public typealias VinceMLImage = NSImage
#endif
//
//  VinceML.swift
//  VinceML
//
//  Created by Vince Carlo Santos on 8/13/25.
//

import Foundation

/// VinceML - A comprehensive Swift package for machine learning model management and training data handling
///
/// This package provides a complete suite of tools for managing machine learning workflows including:
/// - **MLModelService**: Training and inference with Core ML models
/// - **ModelManager**: Model lifecycle management and storage
/// - **TrainingDataService**: Training data organization and management
///
/// ## Key Features
/// - **Easy Model Training**: Simple APIs for training image classification models
/// - **Model Management**: Automatic model storage, loading, and organization
/// - **Training Data Organization**: Directory-based training data management compatible with CreateML
/// - **Cross-Platform**: Supports iOS 15+ and macOS 12+
///
/// ## Quick Start
/// ```swift
/// import VinceML
///
/// // Initialize services
/// let modelService = MLModelService()
/// let modelManager = ModelManager()
/// let trainingDataService = TrainingDataService()
///
/// // Create and train a model
/// try await modelManager.createEmptyModel(name: "MyClassifier")
/// try await trainingDataService.saveTrainingImage(image, with: "Category1", for: "MyClassifier")
/// 
/// let trainingURL = trainingDataService.getTrainingImagesURL(for: "MyClassifier")
/// let model = try await modelService.trainAndSaveModel(from: trainingURL, to: modelURL)
/// ```
///
/// ## Services Overview
///
/// ### MLModelService
/// Handles model training and inference operations:
/// - Training image classification models using CreateML
/// - Performing image classification with trained models
/// - Automatic data validation and augmentation
///
/// ### ModelManager
/// Manages model storage and lifecycle:
/// - Model storage in organized directory structure
/// - Model loading and compilation
/// - Model selection and deletion
///
/// ### TrainingDataService
/// Organizes training data for optimal ML workflows:
/// - Directory-based label organization
/// - Image storage with automatic compression
/// - Training data retrieval and management
///
public struct VinceML {
    /// Current version of the VinceML package
    public static let version = "1.0.0"
    
    /// Package information and metadata
    public static let info = PackageInfo()
}

/// Package information and metadata
public struct PackageInfo {
    public let name = "VinceML"
    public let version = "1.0.0"
    public let description = "A comprehensive Swift package for machine learning model management and training"
    public let author = "Vince Carlo Santos"
    public let platforms = ["iOS 15+", "macOS 12+"]
    
    public var summary: String {
        """
        \(name) v\(version)
        \(description)
        
        Supported Platforms: \(platforms.joined(separator: ", "))
        Author: \(author)
        """
    }
}

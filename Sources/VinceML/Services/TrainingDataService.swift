//
//  TrainingDataService.swift
//  VinceML
//
//  Created by Vince Carlo Santos on 8/13/25.
//

import Foundation
// Platform-agnostic image import
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#else
#error("Unsupported platform: VinceML requires UIKit or AppKit for image handling.")
#endif

// MARK: - Protocol Definition

/// Service protocol for managing training data and label organization
///
/// Provides comprehensive training data management for machine learning models
/// including image storage, label organization, and data retrieval operations.
public protocol TrainingDataServiceProtocol {
    /// Saves a training image with associated label for a specific model
    func saveTrainingImage(_ image: VinceMLImage, with label: String, for modelName: String) async throws
    
    /// Retrieves all training images for a model with metadata
    func getTrainingImages(for modelName: String) async throws -> [TrainingImage]
    
    /// Gets available labels from existing training data
    func getAvailableLabels(for modelName: String) async throws -> Set<String>
    
    /// Gets all labels including manually added ones (same as available labels in current implementation)
    func getAllLabels(for modelName: String) async throws -> Set<String>
    
    /// Adds a new label category for a model
    func addLabel(_ label: String, for modelName: String) async throws
    
    /// Gets the base URL for training images directory
    func getTrainingImagesURL(for modelName: String) -> URL
    
    /// Deletes a specific training image by ID
    func deleteTrainingImage(id: UUID, for modelName: String) async throws
    
    /// Deletes all images for a specific label category
    func deleteAllImagesForLabel(_ label: String, for modelName: String) async throws
}

// MARK: - Service Implementation

/// TrainingDataService manages training data storage and organization for ML models
///
/// This service handles all aspects of training data management including:
/// - Image storage with automatic organization by labels
/// - Directory structure management for CoreML compatibility
/// - Training image metadata and retrieval
/// - Label category management
///
/// **Storage Structure:**
/// ```
/// Documents/VinceML_Models/ModelName/Images/
///   ├── Label1/
///   │   ├── image1.jpg
///   │   ├── image2.jpg
///   │   └── ...
///   ├── Label2/
///   │   ├── image1.jpg
///   │   └── ...
///   └── ...
/// ```
///
/// **CoreML Integration:**
/// The directory structure is designed for direct compatibility with
/// CreateML's `MLImageClassifier(trainingData: .labeledDirectories(at: url))`
/// training method, eliminating the need for data format conversion.
///
/// **Data Consistency:**
/// - Images are stored as JPEG with 80% compression for size optimization
/// - Deterministic UUID generation ensures stable image identification
/// - Directory-based labels eliminate need for separate metadata files
///
/// **Example Usage:**
/// ```swift
/// import VinceML
/// 
/// let service = TrainingDataService()
/// 
/// // Add training images
/// try await service.saveTrainingImage(image, with: "Aviator", for: "SunglassesModel")
/// try await service.saveTrainingImage(image, with: "Wayfarer", for: "SunglassesModel")
/// 
/// // Retrieve for training
/// let trainingURL = service.getTrainingImagesURL(for: "SunglassesModel")
/// // Use trainingURL directly with CreateML
/// ```
public class TrainingDataService: TrainingDataServiceProtocol, ObservableObject {
    
    private let fileManager: FileManager
    private let documentsDirectory: URL
    private let modelsDirectory: URL
    
    /// Initialize the TrainingDataService
    /// - Parameter fileManager: FileManager instance for file operations (default: .default)
    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
        self.documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        self.modelsDirectory = documentsDirectory.appendingPathComponent("VinceML_Models")
        
        // Ensure base models directory exists
        try? fileManager.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
    }
    
    // MARK: - Helper Methods
    
    /// Gets the model directory for a specific model
    /// - Parameter modelName: Name of the model
    /// - Returns: URL to the model's directory
    private func getModelDirectory(for modelName: String) -> URL {
        return modelsDirectory.appendingPathComponent(modelName)
    }
    
    /// Gets the images directory for a specific model
    /// - Parameter modelName: Name of the model
    /// - Returns: URL to the model's training images directory
    private func getImagesDirectory(for modelName: String) -> URL {
        return getModelDirectory(for: modelName).appendingPathComponent("Images")
    }
    
    /// Saves a training image with automatic organization by label
    ///
    /// Stores the image in the appropriate label subdirectory with automatic
    /// directory creation and optimized compression settings.
    ///
    /// **Process:**
    /// 1. Creates model and label directories if they don't exist
    /// 2. Generates unique filename to prevent conflicts
    /// 3. Compresses image to JPEG format (80% quality)
    /// 4. Saves to label-specific subdirectory
    ///
    /// **Directory Structure Created:**
    /// `VinceML_Models/ModelName/Images/LabelName/UUID.jpg`
    ///
    /// - Parameters:
    ///   - image: UIImage to save as training data
    ///   - label: Classification label for this image
    ///   - modelName: Name of the model this training data belongs to
    /// - Throws: TrainingDataError.imageConversionFailed if image cannot be converted to JPEG
    public func saveTrainingImage(_ image: VinceMLImage, with label: String, for modelName: String) async throws {
    // Ensure complete directory structure exists
    let imagesDirectory = getImagesDirectory(for: modelName)
    try fileManager.createDirectory(at: imagesDirectory, withIntermediateDirectories: true)
    let labelDirectory = imagesDirectory.appendingPathComponent(label)
    try fileManager.createDirectory(at: labelDirectory, withIntermediateDirectories: true)
    let fileName = "\(UUID().uuidString).jpg"
    let imageURL = labelDirectory.appendingPathComponent(fileName)
#if canImport(UIKit)
    guard let imageData = image.jpegData(compressionQuality: 0.8) else {
        throw TrainingDataError.imageConversionFailed
    }
#elseif canImport(AppKit)
    guard let tiffData = image.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: tiffData),
          let imageData = bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.8]) else {
        throw TrainingDataError.imageConversionFailed
    }
#else
    throw TrainingDataError.imageConversionFailed
#endif
    try imageData.write(to: imageURL)
    }
    
    /// Retrieves all training images for a model with complete metadata
    ///
    /// Scans the model's training data directory structure and builds
    /// a comprehensive list of all training images with their associated
    /// labels and metadata.
    ///
    /// **Process:**
    /// 1. Scans all label subdirectories in the model's Images folder
    /// 2. Identifies JPEG files in each label directory
    /// 3. Creates TrainingImage records with deterministic IDs
    /// 4. Returns sorted list for consistent UI presentation
    ///
    /// - Parameter modelName: Name of the model to retrieve training images for
    /// - Returns: Array of TrainingImage records with metadata
    /// - Throws: File system errors if directory scanning fails
    public func getTrainingImages(for modelName: String) async throws -> [TrainingImage] {
        let imagesDirectory = getModelDirectory(for: modelName).appendingPathComponent("Images")
        
        // Return empty array if no training data exists yet
        guard fileManager.fileExists(atPath: imagesDirectory.path) else {
            return []
        }

        var trainingImages: [TrainingImage] = []
        
        // Scan all label directories for training images
        let labelDirectories = try fileManager.contentsOfDirectory(at: imagesDirectory, includingPropertiesForKeys: [.isDirectoryKey])
        
        for labelDirectory in labelDirectories {
            // Only process actual directories (label folders)
            guard labelDirectory.hasDirectoryPath else { continue }
            
            let labelName = labelDirectory.lastPathComponent
            let imageFiles = try fileManager.contentsOfDirectory(at: labelDirectory, includingPropertiesForKeys: [.fileSizeKey, .creationDateKey])
            
            // Process JPEG files in this label directory
            for imageFile in imageFiles {
                guard imageFile.pathExtension.lowercased() == "jpg" else { continue }
                
                let trainingImage = TrainingImage(label: labelName, fileName: imageFile.lastPathComponent)
                trainingImages.append(trainingImage)
            }
        }
        
        // Return sorted list for consistent UI presentation
        return trainingImages.sorted { $0.label < $1.label }
    }
    
    /// Gets available labels from existing training data
    ///
    /// Scans the model's Images directory to identify all label categories
    /// that currently contain training data. Labels are derived from the
    /// directory structure rather than stored metadata.
    ///
    /// **Directory-Based Labels:**
    /// This approach ensures labels are always in sync with actual training
    /// data and are automatically compatible with CreateML's expected format.
    ///
    /// - Parameter modelName: Name of the model to get labels for
    /// - Returns: Set of label names currently used in training data
    /// - Throws: File system errors if directory scanning fails
    public func getAvailableLabels(for modelName: String) async throws -> Set<String> {
        let imagesDirectory = getImagesDirectory(for: modelName)
        
        guard fileManager.fileExists(atPath: imagesDirectory.path) else {
            return Set<String>()
        }
        
        // Get all subdirectories (label folders) in the Images directory
        let contents = try fileManager.contentsOfDirectory(at: imagesDirectory, includingPropertiesForKeys: [.isDirectoryKey])
        
        let labelDirectories = contents.filter { url in
            let isDirectory = try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory
            return isDirectory == true
        }
        
        // Return sorted set of label names for consistent UI presentation
        return Set(labelDirectories.map { $0.lastPathComponent })
    }
    
    /// Gets all labels including manually added ones
    ///
    /// In the current implementation, this returns the same as `getAvailableLabels`
    /// since all labels are directory-based. This method exists for API compatibility
    /// and future extensions that might support pre-defined label sets.
    ///
    /// - Parameter modelName: Name of the model to get labels for
    /// - Returns: Set of all label names (same as available labels)
    /// - Throws: File system errors if directory scanning fails
    public func getAllLabels(for modelName: String) async throws -> Set<String> {
        // Current implementation uses only directory-based labels
        // No separate label storage mechanism needed
        return try await getAvailableLabels(for: modelName)
    }
    
    /// Adds a new label category for a model
    ///
    /// Creates a new label directory in the model's training data structure.
    /// This prepares the label for receiving training images and ensures
    /// it appears in label lists even before images are added.
    ///
    /// **Use Cases:**
    /// - Pre-defining label categories during model setup
    /// - Ensuring consistent label naming across training sessions
    /// - UI preparation for label selection interfaces
    ///
    /// - Parameters:
    ///   - label: Name of the new label category
    ///   - modelName: Name of the model to add the label to
    /// - Throws: File system errors if directory creation fails
    public func addLabel(_ label: String, for modelName: String) async throws {
        guard !label.isEmpty else { return }
        
        // Create the label directory in the model's Images folder
        let imagesDirectory = getImagesDirectory(for: modelName)
        let labelDirectory = imagesDirectory.appendingPathComponent(label)
        
        // Create directory structure if it doesn't exist
        if !fileManager.fileExists(atPath: labelDirectory.path) {
            try fileManager.createDirectory(at: labelDirectory, withIntermediateDirectories: true)
        }
    }
    
    /// Gets the base URL for training images directory
    ///
    /// Provides the root directory containing all training data for a model.
    /// This URL can be used directly with CreateML for training operations.
    ///
    /// **CreateML Integration:**
    /// The returned URL is structured for direct use with:
    /// `MLImageClassifier(trainingData: .labeledDirectories(at: url))`
    ///
    /// - Parameter modelName: Name of the model
    /// - Returns: URL to the training images directory
    public func getTrainingImagesURL(for modelName: String) -> URL {
        return getImagesDirectory(for: modelName)
    }
    
    /// Deletes a specific training image by ID
    ///
    /// Removes a single training image from storage using its unique identifier.
    /// Uses the deterministic ID system to locate the exact file without
    /// requiring database lookups.
    ///
    /// **Process:**
    /// 1. Scans training images to find matching ID
    /// 2. Locates corresponding file in label directory
    /// 3. Removes file from storage
    ///
    /// - Parameters:
    ///   - id: Unique identifier of the training image to delete
    ///   - modelName: Name of the model containing the image
    /// - Throws: TrainingDataError.imageNotFound if image doesn't exist
    public func deleteTrainingImage(id: UUID, for modelName: String) async throws {
        let trainingImages = try await getTrainingImages(for: modelName)
        
        guard let imageToDelete = trainingImages.first(where: { $0.id == id }) else {
            throw TrainingDataError.imageNotFound
        }
        
        let imageURL = getImagesDirectory(for: modelName)
            .appendingPathComponent(imageToDelete.label)
            .appendingPathComponent(imageToDelete.fileName)
        
        // Remove the specific image file
        try fileManager.removeItem(at: imageURL)
    }
    
    /// Deletes all images for a specific label category
    ///
    /// Removes the entire label directory and all training images within it.
    /// This effectively removes the label category from the model's training data.
    ///
    /// **Use Cases:**
    /// - Removing incorrect label categories
    /// - Cleaning up unused labels
    /// - Resetting specific categories during model refinement
    ///
    /// **Warning:** This operation cannot be undone and will remove all
    /// training images associated with the specified label.
    ///
    /// - Parameters:
    ///   - label: Name of the label category to delete
    ///   - modelName: Name of the model containing the label
    /// - Throws: File system errors if deletion fails
    public func deleteAllImagesForLabel(_ label: String, for modelName: String) async throws {
        let labelDirectory = getImagesDirectory(for: modelName).appendingPathComponent(label)
        
        if fileManager.fileExists(atPath: labelDirectory.path) {
            try fileManager.removeItem(at: labelDirectory)
        }
    }    
}

// MARK: - Service Errors
public enum TrainingDataError: LocalizedError {
    case imageConversionFailed
    case imageNotFound
    case directoryCreationFailed
    
    public var errorDescription: String? {
        switch self {
        case .imageConversionFailed:
            return "Failed to convert image to data"
        case .imageNotFound:
            return "Training image not found"
        case .directoryCreationFailed:
            return "Failed to create directory"
        }
    }
}

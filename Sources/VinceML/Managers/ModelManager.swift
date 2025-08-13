//
//  ModelManager.swift
//  VinceML
//
//  Created by Vince Carlo Santos on 8/13/25.
//

import Foundation
import CoreML
import CreateML
import Vision

// MARK: - Protocol Definition

/// Protocol defining model management operations for the application
///
/// Provides comprehensive model lifecycle management including storage,
/// loading, deletion, and selection of machine learning models.
public protocol ModelManagerProtocol {
    /// Retrieves the currently selected model for inference
    func getCurrentModel() async -> MLModel?
    
    /// Saves a model from an external URL to app storage
    func saveModelFromURL(_ sourceURL: URL, name: String) async throws
    
    /// Saves a trained model to app storage with proper compilation
    func saveTrainedModel(from trainedModelURL: URL, name: String) async throws
    
    /// Loads a specific model by name from storage
    func loadModel(name: String) async throws -> MLModel?
    
    /// Gets list of all available model names in storage
    func getAvailableModels() async -> [String]
    
    /// Deletes a model and all associated data from storage
    func deleteModel(name: String) async throws
    
    /// Gets the file URL for a compiled model
    func getModelURL(name: String) -> URL
    
    /// Gets the file URL for training data associated with a model
    func getModelTrainingURL(name: String) -> URL
    
    /// Creates a new empty model structure ready for training
    func createEmptyModel(name: String) async throws
    
    /// Gets the name of the currently selected model
    func getSelectedModelName() async -> String?
    
    /// Sets the currently selected model for inference
    func setSelectedModel(name: String) async
    
    /// Cleans up legacy files from previous app versions
    func cleanupLegacyLabelsFiles() async
}

// MARK: - Manager Implementation

/// ModelManager handles all model storage, loading, and lifecycle operations
///
/// This manager provides centralized handling of machine learning models including:
/// - Model storage and organization in app documents directory
/// - Automatic model selection and loading
/// - Training data organization and access
/// - Legacy file cleanup and maintenance
///
/// **Storage Structure:**
/// ```
/// Documents/Models/
///   ├── ModelName1/
///   │   ├── ModelName1.mlmodelc (compiled model)
///   │   ├── Images/ (training data organized by labels)
///   │   │   ├── Label1/
///   │   │   └── Label2/
///   │   └── ModelName1.txt (placeholder for untrained models)
///   └── ModelName2/
///       └── ...
/// ```
///
/// **Model Selection:**
/// - Maintains a selected model preference in UserDefaults
/// - Auto-selects first available model if none selected
/// - Falls back gracefully when selected model is unavailable
///
/// **Example Usage:**
/// ```swift
/// import VinceML
/// 
/// let manager = ModelManager()
/// 
/// // Create and train a model
/// try await manager.createEmptyModel(name: "SunglassesClassifier")
/// try await manager.saveTrainedModel(from: trainedURL, name: "SunglassesClassifier")
/// 
/// // Use the model
/// let currentModel = await manager.getCurrentModel()
/// ```
public class ModelManager: ModelManagerProtocol, ObservableObject {
    
    private let fileManager: FileManager
    private let documentsDirectory: URL
    private let modelsDirectory: URL
    private let selectedModelKey = "VinceML_SelectedModelName"
    
    /// Initialize the ModelManager with proper directory structure
    /// - Parameter fileManager: FileManager instance for file operations (default: .default)
    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
        self.documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        self.modelsDirectory = documentsDirectory.appendingPathComponent("VinceML_Models")
        
        // Ensure models directory structure exists
        try? fileManager.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        
        // Perform maintenance tasks asynchronously
        Task {
            await cleanupLegacyLabelsFiles()
        }
    }
    
    /// Retrieves the currently active model for inference
    ///
    /// Attempts to load the user's selected model, falling back to the first
    /// available model if the selection is invalid. Auto-updates the selection
    /// when falling back to ensure consistency.
    ///
    /// **Selection Priority:**
    /// 1. User's explicitly selected model (from UserDefaults)
    /// 2. First available model (alphabetically)
    /// 3. nil if no models exist
    ///
    /// - Returns: The currently active MLModel, or nil if no models available
    public func getCurrentModel() async -> MLModel? {
        // Try to load user's selected model first
        if let selectedModelName = await getSelectedModelName() {
            if let model = try? await loadModel(name: selectedModelName) {
                return model
            }
        }
        
        // Fall back to first available model if selection is invalid
        let availableModels = await getAvailableModels()
        
        if let firstModel = availableModels.first {
            await setSelectedModel(name: firstModel)
            if let model = try? await loadModel(name: firstModel) {
                return model
            }
        }
        
        return nil
    }
    
    /// Saves a model from an external URL to app storage
    ///
    /// Copies a model bundle from an external source (e.g., app bundle, downloads)
    /// to the app's managed storage area. Automatically selects the saved model
    /// as the current active model.
    ///
    /// - Parameters:
    ///   - sourceURL: URL of the source model bundle to copy
    ///   - name: Name to assign to the saved model
    /// - Throws: File operation errors if copy fails
    public func saveModelFromURL(_ sourceURL: URL, name: String) async throws {
        let modelURL = getModelURL(name: name)
        
        // Remove existing model if present
        if fileManager.fileExists(atPath: modelURL.path) {
            try fileManager.removeItem(at: modelURL)
        }
        
        // Copy model bundle to managed storage
        try fileManager.copyItem(at: sourceURL, to: modelURL)
        
        // Auto-select as current model
        await setSelectedModel(name: name)
    }
    
    /// Saves a trained model to storage with proper compilation
    ///
    /// Handles the complete process of saving a newly trained model including:
    /// - Automatic compilation from .mlmodel to .mlmodelc if needed
    /// - Proper directory structure creation
    /// - Cleanup of temporary files and placeholders
    /// - Model selection update
    ///
    /// **File Handling:**
    /// - Compiles .mlmodel files to optimized .mlmodelc format
    /// - Removes placeholder files created during model initialization
    /// - Cleans up temporary training files
    ///
    /// - Parameters:
    ///   - trainedModelURL: URL of the trained model file
    ///   - name: Name to assign to the saved model
    /// - Throws: ModelManagerError or file operation errors
    public func saveTrainedModel(from trainedModelURL: URL, name: String) async throws {
        let modelDirectory = modelsDirectory.appendingPathComponent(name)
        let finalModelURL = getModelURL(name: name)
        
        // Compile .mlmodel to optimized .mlmodelc format if needed
        let compiledURL: URL
        if trainedModelURL.pathExtension == "mlmodel" {
            compiledURL = try MLModel.compileModel(at: trainedModelURL)
        } else {
            compiledURL = trainedModelURL
        }
        
        // Ensure model directory structure exists
        try fileManager.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
        
        // Install compiled model in final location
        if fileManager.fileExists(atPath: finalModelURL.path) {
            try fileManager.removeItem(at: finalModelURL)
        }
        
        try fileManager.copyItem(at: compiledURL, to: finalModelURL)
        
        // Clean up placeholder file from model initialization
        let placeholderURL = modelDirectory.appendingPathComponent("\(name).txt")
        if fileManager.fileExists(atPath: placeholderURL.path) {
            try fileManager.removeItem(at: placeholderURL)
        }
        
        // Clean up original .mlmodel file if it's in our model directory
        if trainedModelURL.pathExtension == "mlmodel" && 
           trainedModelURL.deletingLastPathComponent() == modelDirectory {
            try? fileManager.removeItem(at: trainedModelURL)
        }
        
        // Set as currently active model
        await setSelectedModel(name: name)
    }

    /// Loads a specific model by name from storage
    ///
    /// Attempts to load and instantiate an MLModel from the compiled
    /// model file in storage. Returns nil if the model doesn't exist.
    ///
    /// - Parameter name: Name of the model to load
    /// - Returns: MLModel instance if found and loadable, nil otherwise
    /// - Throws: CoreML errors if model compilation or loading fails
    public func loadModel(name: String) async throws -> MLModel? {
        let modelURL = getModelURL(name: name)
        
        guard fileManager.fileExists(atPath: modelURL.path) else {
            return nil
        }
        
        return try MLModel(contentsOf: modelURL)
    }
    
    /// Gets list of all available model names in storage
    ///
    /// Scans the models directory for valid model folders and returns
    /// their names. Only includes directories that represent model storage.
    ///
    /// - Returns: Array of model names currently available in storage
    public func getAvailableModels() async -> [String] {
        do {
            let contents = try fileManager.contentsOfDirectory(at: modelsDirectory, includingPropertiesForKeys: [.isDirectoryKey])
            return contents
                .filter { url in
                    // Only include actual directories (model folders)
                    let isDirectory = try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory
                    return isDirectory == true
                }
                .map { $0.lastPathComponent }
                .sorted() // Return in consistent alphabetical order
        } catch {
            return []
        }
    }
    
    /// Deletes a model and all associated data from storage
    ///
    /// Completely removes a model including:
    /// - Compiled model file (.mlmodelc)
    /// - Training data directory and images
    /// - Model directory structure
    /// - Selection preference if this was the selected model
    ///
    /// - Parameter name: Name of the model to delete
    /// - Throws: File operation errors if deletion fails
    public func deleteModel(name: String) async throws {
        let modelDirectory = modelsDirectory.appendingPathComponent(name)
        
        if fileManager.fileExists(atPath: modelDirectory.path) {
            try fileManager.removeItem(at: modelDirectory)
        }
        
        // Clear model selection if this was the selected model
        if await getSelectedModelName() == name {
            UserDefaults.standard.removeObject(forKey: selectedModelKey)
        }
    }
    
    /// Gets the file URL for a compiled model
    /// - Parameter name: Name of the model
    /// - Returns: URL to the compiled .mlmodelc file
    public func getModelURL(name: String) -> URL {
        return modelsDirectory.appendingPathComponent(name).appendingPathComponent("\(name).mlmodelc")
    }
    
    /// Gets the file URL for training data associated with a model
    /// - Parameter name: Name of the model
    /// - Returns: URL to the .mlmodel file used during training
    public func getModelTrainingURL(name: String) -> URL {
        return modelsDirectory.appendingPathComponent(name).appendingPathComponent("\(name).mlmodel")
    }
    
    /// Creates a new empty model structure ready for training
    ///
    /// Sets up the complete directory structure for a new model including:
    /// - Model directory with the specified name
    /// - Images subdirectory for training data organization
    /// - Placeholder file with model information and status
    ///
    /// **Directory Structure Created:**
    /// ```
    /// VinceML_Models/ModelName/
    ///   ├── Images/ (for training data organized by labels)
    ///   └── ModelName.txt (placeholder until training completes)
    /// ```
    ///
    /// - Parameter name: Name for the new model
    /// - Throws: ModelManagerError.modelAlreadyExists if model name is taken
    public func createEmptyModel(name: String) async throws {
        let modelDirectory = modelsDirectory.appendingPathComponent(name)
        let placeholderURL = modelDirectory.appendingPathComponent("\(name).txt")
        
        // Ensure model name is available
        if fileManager.fileExists(atPath: modelDirectory.path) {
            throw ModelManagerError.modelAlreadyExists
        }
        
        // Create complete directory structure for new model
        try fileManager.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
        
        // Create training images directory
        let imagesDirectory = modelDirectory.appendingPathComponent("Images")
        try fileManager.createDirectory(at: imagesDirectory, withIntermediateDirectories: true)
        
        // Create informational placeholder file
        let modelInfo = """
        VinceML Model: \(name)
        Created: \(Date())
        Status: Empty - Ready for Training
        Type: Image Classifier
        Package Version: \(VinceML.version)
        
        Directory Structure:
        - \(name).mlmodelc (will be created after training)
        - Images/ (training images organized by label folders)
        
        Training Process:
        1. Add training images organized by label in Images/ directory
        2. Ensure minimum 5 images per label category
        3. Train model using VinceML MLModelService
        4. Trained model will replace this placeholder file
        """
        
        try modelInfo.write(to: placeholderURL, atomically: true, encoding: .utf8)
    }
    
    /// Gets the name of the currently selected model
    /// - Returns: Selected model name from UserDefaults, or nil if none selected
    public func getSelectedModelName() async -> String? {
        return UserDefaults.standard.string(forKey: selectedModelKey)
    }
    
    /// Sets the currently selected model for inference
    /// - Parameter name: Name of the model to select
    public func setSelectedModel(name: String) async {
        UserDefaults.standard.set(name, forKey: selectedModelKey)
    }
    
    /// Cleans up legacy files from previous app versions
    ///
    /// Removes outdated labels.json files that were used in earlier versions
    /// of the app. The current version uses directory structure instead of
    /// JSON files for label organization.
    ///
    /// **Background:** Previous versions stored labels in JSON files alongside
    /// model data. Current version uses folder structure in Images/ directory
    /// for more intuitive organization and CoreML compatibility.
    public func cleanupLegacyLabelsFiles() async {
        let models = await getAvailableModels()
        
        for modelName in models {
            let modelDirectory = modelsDirectory.appendingPathComponent(modelName)
            let labelsFileURL = modelDirectory.appendingPathComponent("labels.json")
            
            // Remove legacy labels.json files if they exist
            if fileManager.fileExists(atPath: labelsFileURL.path) {
                do {
                    try fileManager.removeItem(at: labelsFileURL)
                    // Note: Removed debug print - cleanup is now silent
                } catch {
                    // Note: Removed debug print - errors are handled silently
                    // Legacy file cleanup failures are non-critical
                }
            }
        }
    }
}

// MARK: - Manager Errors
public enum ModelManagerError: LocalizedError {
    case modelNotFound
    case saveFailed
    case compilationFailed
    case modelAlreadyExists
    
    public var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "Model not found"
        case .saveFailed:
            return "Failed to save model - use saveModelFromURL or saveTrainedModel instead"
        case .compilationFailed:
            return "Failed to compile model"
        case .modelAlreadyExists:
            return "A model with this name already exists"
        }
    }
}

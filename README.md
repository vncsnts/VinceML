# VinceML


A comprehensive Swift package for machine learning model management and training data handling on all Apple platforms (iOS, macOS, tvOS, watchOS).

[![Swift Version](https://img.shields.io/badge/Swift-5.7+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-iOS%2015%2B%20%7C%20macOS%2012%2B%20%7C%20tvOS%2015%2B%20%7C%20watchOS%208%2B-blue.svg)](https://developer.apple.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Features

VinceML provides a complete suite of tools for managing machine learning workflows including:

- **🧠 MLModelService**: Training and inference with Core ML models
- **📁 ModelManager**: Model lifecycle management and storage  
- **🖼️ TrainingDataService**: Training data organization and management
- **📱 Cross-Platform**: Supports iOS 15+, macOS 12+, tvOS 15+, and watchOS 8+
- **🔧 Easy Integration**: Simple APIs with comprehensive documentation

## Installation

### Swift Package Manager

Add VinceML to your project using Xcode:

1. Go to **File** → **Add Package Dependencies**
2. Enter the repository URL: `https://github.com/vncsnts/VinceML.git`
3. Select the version and add to your target

Or add it to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/vncsnts/VinceML.git", from: "1.0.0")
]
```

## Quick Start

```swift
import VinceML

// Initialize services
let modelService = MLModelService()
let modelManager = ModelManager()
let trainingDataService = TrainingDataService()

// Create and set up a new model
try await modelManager.createEmptyModel(name: "MyClassifier")

// Add training images

// Use VinceMLImage for platform-agnostic image handling
// VinceMLImage is a typealias for UIImage (iOS/tvOS/watchOS) or NSImage (macOS)
try await trainingDataService.saveTrainingImage(image1, with: "Category1", for: "MyClassifier")
try await trainingDataService.saveTrainingImage(image2, with: "Category2", for: "MyClassifier")

// Train the model
let trainingURL = trainingDataService.getTrainingImagesURL(for: "MyClassifier")
let modelURL = modelManager.getModelTrainingURL(name: "MyClassifier")
let trainedModel = try await modelService.trainAndSaveModel(from: trainingURL, to: modelURL)

// Save and use the trained model
try await modelManager.saveTrainedModel(from: modelURL, name: "MyClassifier")
let results = await modelService.classifyImage(testImage, using: trainedModel)
```

## Services Overview

### MLModelService

Handles model training and inference operations:

- **Training**: Create image classification models using CreateML
- **Inference**: Perform image classification with trained models  
- **Validation**: Automatic data validation and augmentation
- **Results**: Top 3 predictions with confidence percentages

```swift
let service = MLModelService()

// Train a model
let model = try await service.trainAndSaveModel(
    from: trainingDataURL,
    to: modelSaveURL
)

// Classify images (use VinceMLImage)
let results = await service.classifyImage(image, using: model)
// Returns: ["Cat: 95.67%", "Dog: 3.21%", "Bird: 1.12%"]
```

### ModelManager

Manages model storage and lifecycle:

- **Storage**: Organized directory structure in app documents
- **Loading**: Automatic model compilation and loading
- **Selection**: Current model selection and management
- **Cleanup**: Legacy file cleanup and maintenance

```swift
let manager = ModelManager()

// Create and manage models
try await manager.createEmptyModel(name: "PetClassifier")
let availableModels = await manager.getAvailableModels()
let currentModel = await manager.getCurrentModel()

// Delete models
try await manager.deleteModel(name: "OldModel")
```

### TrainingDataService

Organizes training data for optimal ML workflows:

- **Organization**: Directory-based label organization
- **Storage**: Image storage with automatic compression
- **Compatibility**: Direct CreateML integration
- **Management**: Training data retrieval and cleanup

```swift
let service = TrainingDataService()

// Add training data
try await service.saveTrainingImage(image, with: "Label1", for: "MyModel") // image is VinceMLImage

// Retrieve training information
let images = try await service.getTrainingImages(for: "MyModel")
let labels = try await service.getAvailableLabels(for: "MyModel")

// Get training URL for CreateML
let trainingURL = service.getTrainingImagesURL(for: "MyModel")
```

## Storage Structure

VinceML uses a well-organized directory structure that's compatible with CreateML:

```
Documents/VinceML_Models/
├── ModelName1/
│   ├── ModelName1.mlmodelc          # Compiled model
│   ├── Images/                      # Training data
│   │   ├── Label1/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   │   ├── Label2/
│   │   │   ├── image1.jpg
│   │   │   └── ...
│   │   └── ...
│   └── ModelName1.txt               # Placeholder for untrained models
└── ModelName2/
    └── ...
```

## Requirements

- iOS 15.0+ / macOS 12.0+ / tvOS 15.0+ / watchOS 8.0+
- Xcode 14.0+
- Swift 5.7+

## Core ML Integration

VinceML is designed to work seamlessly with Apple's Core ML and CreateML frameworks:

- **Training**: Uses `MLImageClassifier` with automatic data augmentation
- **Storage**: Compiled `.mlmodelc` format for optimal performance
- **Inference**: Vision framework integration for efficient classification
- **Compatibility**: Standard Core ML model format for interoperability

## Example App Integration

See the [VinceMLApp](https://github.com/vncsnts/VinceMLApp) project for a complete example of VinceML integration in a real iOS app.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

VinceML is available under the MIT license. See the LICENSE file for more info.

## Author

Created by [Vince Carlo Santos](https://github.com/vncsnts)

---

**VinceML** - Simplifying machine learning workflows on all Apple platforms.

//
//  TrainingImage.swift
//  VinceML
//
//  Created by Vince Carlo Santos on 8/13/25.
//

import Foundation

// MARK: - Data Models

/// Represents a training image with metadata for machine learning model training
///
/// Provides a structured representation of training images including
/// identification, categorization, and file system information.
///
/// **ID Generation:**
/// Uses deterministic UUID generation based on label and filename to ensure
/// the same file always receives the same ID, enabling consistent references
/// across app sessions without requiring persistent storage.
public struct TrainingImage: Identifiable, Codable {
    /// Unique identifier for this training image (deterministic based on label/filename)
    public let id: UUID
    
    /// Classification label/category for this image
    public let label: String
    
    /// Filename of the image file in storage
    public let fileName: String
    
    /// Timestamp when this training image was created
    public let dateCreated: Date
    
    /// Initialize a new training image record
    /// - Parameters:
    ///   - label: Classification label for the image
    ///   - fileName: Name of the image file in storage
    public init(label: String, fileName: String) {
        // Generate deterministic UUID from label and filename
        // This ensures consistent IDs across app launches without database storage
        let combinedString = "\(label)/\(fileName)"
        self.id = UUID(uuidString: combinedString.deterministic5()) ?? UUID()
        self.label = label
        self.fileName = fileName
        self.dateCreated = Date()
    }
}

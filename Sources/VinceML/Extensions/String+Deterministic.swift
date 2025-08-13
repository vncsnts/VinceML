//
//  String+Deterministic.swift
//  VinceML
//
//  Created by Vince Carlo Santos on 8/13/25.
//

import Foundation

/// Extension for deterministic UUID generation from strings
public extension String {
    /// Generates a deterministic UUID-like string from the input string
    ///
    /// Creates consistent UUID strings from the same input, enabling
    /// stable identification without persistent storage requirements.
    ///
    /// **Note:** This is a simplified approach for app-local use.
    /// For production systems requiring true uniqueness guarantees,
    /// consider using cryptographic hash functions.
    ///
    /// - Returns: UUID-formatted string derived from input string
    func deterministic5() -> String {
        let hash = self.hash
        let uuidString = String(format: "%08X-0000-0000-0000-%012X", abs(hash) & 0xFFFFFFFF, abs(hash) >> 32)
        return uuidString
    }
}

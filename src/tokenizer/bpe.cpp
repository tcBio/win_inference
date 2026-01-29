#include "tokenizer/tokenizer.hpp"

#include <map>
#include <limits>
#include <vector>
#include <string>
#include <utility>

// BPE implementation details
// This file contains the byte-pair encoding algorithm implementation
// For Qwen2.5, the tokenizer uses a tiktoken-style BPE

namespace qwen::tokenizer::bpe {

/// Byte-pair encoding merge operation
struct BPEMerge {
    std::string first;
    std::string second;
    int rank;
};

/// Get the pairs of adjacent symbols in a word
std::vector<std::pair<std::string, std::string>> get_pairs(
    const std::vector<std::string>& word) {

    std::vector<std::pair<std::string, std::string>> pairs;
    if (word.size() < 2) return pairs;

    for (size_t i = 0; i < word.size() - 1; ++i) {
        pairs.emplace_back(word[i], word[i + 1]);
    }
    return pairs;
}

/// Apply BPE merges to a word
std::vector<std::string> apply_bpe(
    const std::string& token,
    const std::map<std::pair<std::string, std::string>, int>& merges) {

    // Start with individual characters
    std::vector<std::string> word;
    for (char c : token) {
        word.push_back(std::string(1, c));
    }

    if (word.size() == 1) {
        return word;
    }

    while (true) {
        auto pairs = get_pairs(word);
        if (pairs.empty()) break;

        // Find the pair with the lowest rank (highest priority)
        std::pair<std::string, std::string> best_pair;
        int best_rank = std::numeric_limits<int>::max();

        for (const auto& pair : pairs) {
            auto it = merges.find(pair);
            if (it != merges.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pair = pair;
            }
        }

        if (best_rank == std::numeric_limits<int>::max()) {
            break;  // No more merges possible
        }

        // Apply the merge
        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) {
            if (i < word.size() - 1 &&
                word[i] == best_pair.first &&
                word[i + 1] == best_pair.second) {
                new_word.push_back(best_pair.first + best_pair.second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }
        word = std::move(new_word);

        if (word.size() == 1) {
            break;
        }
    }

    return word;
}

/// Pre-tokenize text into words (for BPE)
std::vector<std::string> pre_tokenize(const std::string& text) {
    // Simple whitespace-based pre-tokenization
    // Qwen uses a more sophisticated regex-based approach
    std::vector<std::string> tokens;
    std::string current;

    for (char c : text) {
        if (c == ' ' || c == '\n' || c == '\t') {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
            // Include whitespace as a token with special prefix
            tokens.push_back("Ġ" + std::string(1, c == ' ' ? c : ' '));
        } else {
            current += c;
        }
    }

    if (!current.empty()) {
        tokens.push_back(current);
    }

    return tokens;
}

}  // namespace qwen::tokenizer::bpe

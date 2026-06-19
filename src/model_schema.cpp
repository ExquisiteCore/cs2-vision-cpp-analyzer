#include "vision_analyzer/model_schema.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace vision_analyzer {

std::string default_model_schema_path(const std::string& model_path) {
    return model_path + ".schema.json";
}

ModelSchema load_model_schema(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open model schema: " + path);
    }
    const std::string content((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const std::regex classes_regex(R"json("classes"\s*:\s*\[([^\]]*)\])json");
    std::smatch classes_match;
    if (!std::regex_search(content, classes_match, classes_regex)) {
        throw std::runtime_error("model schema is missing classes array: " + path);
    }

    ModelSchema schema;
    const std::string classes_body = classes_match[1].str();
    const std::regex string_regex(R"json("([^"]+)")json");
    for (auto it = std::sregex_iterator(classes_body.begin(), classes_body.end(), string_regex);
         it != std::sregex_iterator();
         ++it) {
        schema.classes.push_back((*it)[1].str());
    }
    return schema;
}

void validate_model_schema(const ModelSchema& schema) {
    const auto& expected = class_names();
    if (schema.classes.size() != expected.size()) {
        throw std::runtime_error("model schema class count mismatch");
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (schema.classes[index] != expected[index]) {
            std::ostringstream message;
            message << "model schema class mismatch at index " << index
                    << ": expected " << expected[index]
                    << ", got " << schema.classes[index];
            throw std::runtime_error(message.str());
        }
    }
}

void validate_configured_model_schema(const Options& options) {
    const bool explicit_schema = !options.model_schema_path.empty();
    const std::string schema_path = explicit_schema
        ? options.model_schema_path
        : default_model_schema_path(options.model_path);
    if (!explicit_schema && !std::filesystem::exists(schema_path)) {
        std::cout << "model_schema=missing path=" << schema_path << '\n';
        return;
    }

    const ModelSchema schema = load_model_schema(schema_path);
    validate_model_schema(schema);
    std::cout << "model_schema=ok path=" << schema_path << " classes=" << schema.classes.size() << '\n';
}

}  // namespace vision_analyzer

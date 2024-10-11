# frozen_string_literal: true

require_relative "lib/wandb/version"

Gem::Specification.new do |spec|
  spec.name = "wandb"
  spec.version = Wandb::VERSION
  spec.authors = ["Brett Shollenberger"]
  spec.email = ["brett.shollenberger@gmail.com"]

  spec.summary = "A Ruby integration for the Weights & Biases platform"
  spec.description = "Log model runs to Weights & Biases"
  spec.homepage = "https://github.com/brettshollenberger/wandb_rb.git"
  spec.license = "MIT"
  spec.files = Dir["lib/**/*.rb"]
  spec.required_ruby_version = ">= 2.5"

  spec.metadata["homepage_uri"] = spec.homepage

  spec.add_dependency "pycall"
end

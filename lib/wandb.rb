require_relative "wandb/version"
require "pry"
require "pycall/import"

# Ensure wandb executable isn't set using Ruby context
ENV["WANDB__EXECUTABLE"] = `which python3`.chomp.empty? ? `which python`.chomp : `which python3`.chomp
py_sys = PyCall.import_module("sys")
py_sys.executable = ENV["WANDB__EXECUTABLE"]

module Wandb
  include PyCall::Import

  class << self
    # Lazy-load the wandb Python module
    def __pyptr__
      @wandb ||= PyCall.import_module("wandb")
    end

    def plot(*args, **kwargs)
      __pyptr__.plot(*args, **kwargs)
    end

    # Expose define_metric
    def define_metric(metric_name, **kwargs)
      __pyptr__.define_metric(name: metric_name.force_encoding("UTF-8"), **kwargs)
    end

    # Expose wandb.Artifact
    def artifact(*args, **kwargs)
      py_artifact = __pyptr__.Artifact.new(*args, **kwargs)
      Artifact.new(py_artifact)
    end

    def error
      __pyptr__.Error
    end

    # Login to Wandb
    def login(api_key: nil, **kwargs)
      kwargs = kwargs.to_h
      kwargs[:key] = api_key if api_key
      __pyptr__.login(**kwargs)
    end

    # Initialize a new run
    def init(**kwargs)
      run = __pyptr__.init(**kwargs)
      @current_run = Run.new(run)
    end

    def latest_run=(run)
      @latest_run = run
    end

    def latest_run
      @latest_run
    end

    # Get the current run
    attr_reader :current_run

    # Log metrics to the current run
    def log(metrics = {})
      raise "No active run. Call Wandb.init() first." unless @current_run

      @current_run.log(metrics.symbolize_keys)
    end

    # Finish the current run
    def finish
      @current_run.finish if @current_run
      @current_run = nil
      __pyptr__.finish
    end

    # Access the Wandb API
    def api
      @api ||= Api.new(__pyptr__.Api.new)
    end

    def plot
      Plot
    end

    def run_url
      raise "No active run. Call Wandb.init() first." unless @current_run

      @current_run.url
    end
  end

  # Run class
  class Run
    def initialize(run)
      @run = run
    end

    def run_id
      @run.run_id
    end

    def log(metrics = {})
      metrics.symbolize_keys!
      @run.log(metrics, {})
    end

    def finish
      @run.finish
    end

    def name
      @run.name
    end

    def name=(new_name)
      @run.name = new_name
    end

    def config
      @run.config
    end

    def config=(new_config)
      @run.config.update(PyCall::Dict.new(new_config))
    end

    def log_artifact(artifact)
      @run.log_artifact(artifact.__pyptr__)
    end

    # Add this new method
    def url
      @run.get_url
    end
  end

  # Artifact class
  class Artifact
    def initialize(artifact)
      @artifact = artifact
    end

    def __pyptr__
      @artifact
    end

    def name
      @artifact.name
    end

    def type
      @artifact.type
    end

    def add_file(local_path, name = nil)
      @artifact.add_file(local_path, name)
    end

    def add_dir(local_dir, name = nil)
      @artifact.add_dir(local_dir, name)
    end

    def get_path(name)
      @artifact.get_path(name)
    end

    def metadata
      @artifact.metadata
    end

    def metadata=(new_metadata)
      @artifact.metadata = new_metadata
    end

    def save
      @artifact.save
    end
  end

  # Api class
  class Api
    def initialize(api)
      @api = api
    end

    def projects(entity = nil)
      projects = @api.projects(entity)
      projects.map { |proj| Project.new(proj) }
    end

    def project(name, entity = nil)
      proj = @api.project(name, entity)
      Project.new(proj)
    end
  end

  # Project class
  class Project
    def initialize(project)
      @project = project
    end

    def name
      @project.name
    end

    def description
      @project.description
    end
  end

  # Table class
  class Table
    attr_accessor :table, :data, :columns

    def initialize(data: {}, columns: [])
      @table = Wandb.__pyptr__.Table.new(data: data, columns: columns)
      @data = data
      @columns = columns
    end

    def __pyptr__
      @table
    end

    def add_data(*args)
      @table.add_data(*args)
    end

    def add_column(name, data)
      @table.add_column(name, data)
    end

    def get_column(name)
      @table.get_column(name)
    end

    def columns
      @table.columns
    end

    def data
      @table.data
    end

    def to_pandas
      @table.get_dataframe
    end
  end

  # Plot class
  class Plot
    class << self
      def bar(table, x_key, y_key, title: nil)
        py_plot = Wandb.__pyptr__.plot.bar(table.__pyptr__, x_key, y_key, title: title)
        new(py_plot)
      end

      def line(table, x_key, y_key, title: nil)
        py_plot = Wandb.__pyptr__.plot.line(table.__pyptr__, x_key, y_key, title: title)
        new(py_plot)
      end

      def scatter(table, x_key, y_key, title: nil)
        py_plot = Wandb.__pyptr__.plot.scatter(table.__pyptr__, x_key, y_key, title: title)
        new(py_plot)
      end

      def histogram(table, value_key, title: nil)
        py_plot = Wandb.__pyptr__.plot.histogram(table.__pyptr__, value_key, title: title)
        new(py_plot)
      end
    end

    def initialize(plot)
      @plot = plot
    end

    def __pyptr__
      @plot
    end
  end
end

require_relative "wandb/xgboost_callback"

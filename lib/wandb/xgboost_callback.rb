require "xgb"
require "tempfile"
require "fileutils"

module Wandb
  class XGBoostCallback < XGBoost::TrainingCallback
    MINIMIZE_METRICS = %w[rmse logloss error] # Add other metrics as needed
    MAXIMIZE_METRICS = %w[auc accuracy] # Add other metrics as needed

    def initialize(log_model: false, log_feature_importance: true, importance_type: "gain", define_metric: true)
      @log_model = log_model
      @log_feature_importance = log_feature_importance
      @importance_type = importance_type
      @define_metric = define_metric

      return if Wandb.current_run

      raise "You must call wandb.init() before WandbCallback()"
    end

    def before_training(model:)
      # Update Wandb config with model configuration
      Wandb.current_run.config = model.params
      Wandb.log(model.params)
      model
    end

    def after_training(model:)
      # Log the model as an artifact
      log_model_as_artifact(model) if @log_model

      # Log feature importance
      log_feature_importance(model) if @log_feature_importance

      # Log best score and best iteration
      return unless model.best_score

      Wandb.log(
        "best_score" => model.best_score.to_f,
        "best_iteration" => model.best_iteration.to_i
      )
    end

    def before_iteration(model:, epoch:, evals:)
      # noop
    end

    def after_iteration(model:, epoch:, evals:, res:)
      res.each do |metric_name, value|
        data, metric = metric_name.split("-", 2)
        full_metric_name = "#{data}-#{metric}"

        if @define_metric
          define_metric(data, metric)
          Wandb.log({ full_metric_name => value })
        else
          Wandb.log({ full_metric_name => value })
        end
      end

      Wandb.log({ "epoch" => epoch })
      @define_metric = false
    end

    private

    def log_model_as_artifact(model)
      Dir.mktmpdir("wandb_xgboost_model") do |tmp_dir|
        model_name = "model.json"
        model_path = File.join(tmp_dir, model_name)
        model.save_model(model_path)

        model_artifact = Wandb.artifact(name: model_name, type: "model")
        model_artifact.add_file(model_path)
        Wandb.current_run.log_artifact(model_artifact)
      end
    end

    def log_feature_importance(model)
      fi = model.score(importance_type: @importance_type)
      fi_data = fi.map { |k, v| [k, v] }

      table = Wandb::Table.new(data: fi_data, columns: %w[Feature Importance])
      bar_plot = Wandb::Plot.bar(table, "Feature", "Importance", title: "Feature Importance")
      Wandb.log({ "Feature Importance" => bar_plot })
    end

    def define_metric(data, metric_name)
      full_metric_name = "#{data}-#{metric_name}"

      if metric_name.downcase.include?("loss") || MINIMIZE_METRICS.include?(metric_name.downcase)
        Wandb.define_metric(full_metric_name, summary: "min")
      elsif MAXIMIZE_METRICS.include?(metric_name.downcase)
        Wandb.define_metric(full_metric_name, summary: "max")
      end
    end
  end
end

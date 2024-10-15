# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Wandb::XGBoostCallback do
  let(:mock_run) { instance_double(Wandb::Run) }
  let(:mock_model) { instance_double(XGBoost::Booster) }

  before do
    allow(Wandb).to receive(:current_run).and_return(mock_run)
    allow(mock_run).to receive(:config=)
    allow(Wandb).to receive(:log)
  end

  describe '#initialize' do
    context 'when Wandb.current_run is not set' do
      before { allow(Wandb).to receive(:current_run).and_return(nil) }

      it 'raises an error' do
        expect { described_class.new }.to raise_error(RuntimeError, "You must call wandb.init() before WandbCallback()")
      end
    end

    context 'when Wandb.current_run is set' do
      it 'initializes without error' do
        expect { described_class.new }.not_to raise_error
      end

      it 'sets default values' do
        callback = described_class.new
        expect(callback.instance_variable_get(:@log_model)).to be false
        expect(callback.instance_variable_get(:@log_feature_importance)).to be true
        expect(callback.instance_variable_get(:@importance_type)).to eq 'gain'
        expect(callback.instance_variable_get(:@define_metric)).to be true
      end

      it 'allows custom values' do
        callback = described_class.new(log_model: true, log_feature_importance: false, importance_type: 'weight', define_metric: false)
        expect(callback.instance_variable_get(:@log_model)).to be true
        expect(callback.instance_variable_get(:@log_feature_importance)).to be false
        expect(callback.instance_variable_get(:@importance_type)).to eq 'weight'
        expect(callback.instance_variable_get(:@define_metric)).to be false
      end
    end
  end

  describe '#before_training' do
    let(:callback) { described_class.new }
    let(:model_params) { { 'max_depth' => 3, 'eta' => 0.1 } }

    before do
      allow(mock_model).to receive(:params).and_return(model_params)
    end

    it 'updates Wandb config with model parameters' do
      expect(mock_run).to receive(:config=).with(model_params)
      expect(Wandb).to receive(:log).with(model_params)
      callback.before_training(model: mock_model)
    end
  end

  describe '#after_training' do
    let(:callback) { described_class.new(log_model: true, log_feature_importance: true) }

    before do
      allow(mock_model).to receive(:best_score).and_return(0.95)
      allow(mock_model).to receive(:best_iteration).and_return(100)
      allow(callback).to receive(:log_model_as_artifact)
      allow(callback).to receive(:log_feature_importance)
    end

    it 'logs the model as an artifact' do
      expect(callback).to receive(:log_model_as_artifact).with(mock_model)
      callback.after_training(model: mock_model)
    end

    it 'logs feature importance' do
      expect(callback).to receive(:log_feature_importance).with(mock_model)
      callback.after_training(model: mock_model)
    end

    it 'logs best score and best iteration' do
      expect(Wandb).to receive(:log).with({ "best_score" => 0.95, "best_iteration" => 100 })
      callback.after_training(model: mock_model)
    end

    context 'when best_score is nil' do
      before { allow(mock_model).to receive(:best_score).and_return(nil) }

      it 'does not log best score and best iteration' do
        expect(Wandb).not_to receive(:log)
        callback.after_training(model: mock_model)
      end
    end
  end

  describe '#before_iteration' do
    let(:callback) { described_class.new }

    it 'does nothing' do
      expect { callback.before_iteration(model: mock_model, epoch: 1, evals: []) }.not_to raise_error
    end
  end

  describe '#after_iteration' do
    let(:callback) { described_class.new }
    let(:res) { { "train-error" => 0.1, "valid-auc" => 0.9 } }

    before do
      allow(callback).to receive(:define_metric)
    end

    it 'logs metrics and epoch' do
      expect(Wandb).to receive(:log).with({ "train-error" => 0.1 })
      expect(Wandb).to receive(:log).with({ "valid-auc" => 0.9 })
      expect(Wandb).to receive(:log).with({ "epoch" => 1 })
      callback.after_iteration(model: mock_model, epoch: 1, evals: [], res: res)
    end

    it 'defines metrics on first iteration' do
      expect(callback).to receive(:define_metric).with("train", "error")
      expect(callback).to receive(:define_metric).with("valid", "auc")
      callback.after_iteration(model: mock_model, epoch: 1, evals: [], res: res)
    end

    it 'does not define metrics on subsequent iterations' do
      callback.after_iteration(model: mock_model, epoch: 1, evals: [], res: res)
      expect(callback).not_to receive(:define_metric)
      callback.after_iteration(model: mock_model, epoch: 2, evals: [], res: res)
    end
  end

  describe '#log_model_as_artifact' do
    let(:callback) { described_class.new(log_model: true) }
    let(:mock_artifact) { instance_double(Wandb::Artifact) }

    before do
      allow(mock_model).to receive(:save_model)
      allow(Wandb).to receive(:artifact).and_return(mock_artifact)
      allow(mock_run).to receive(:log_artifact)
    end

    it 'saves the model in a temp directory and logs it as an artifact' do
      expect(Dir).to receive(:mktmpdir).with("wandb_xgboost_model").and_yield("/tmp/wandb_xgboost_model")
      expect(mock_model).to receive(:save_model).with("/tmp/wandb_xgboost_model/model.json")
      expect(Wandb).to receive(:artifact).with(name: "model.json", type: "model").and_return(mock_artifact)
      expect(mock_artifact).to receive(:add_file).with("/tmp/wandb_xgboost_model/model.json")
      expect(mock_run).to receive(:log_artifact).with(mock_artifact)

      callback.send(:log_model_as_artifact, mock_model)
    end
  end

  describe '#log_feature_importance' do
    let(:callback) { described_class.new(log_feature_importance: true) }
    let(:mock_table) { instance_double(Wandb::Table) }
    let(:mock_plot) { instance_double(Wandb::Plot) }

    before do
      allow(mock_model).to receive(:score).and_return({ 'feature1' => 0.5, 'feature2' => 0.3 })
      allow(Wandb::Table).to receive(:new).and_return(mock_table)
      allow(Wandb::Plot).to receive(:bar).and_return(mock_plot)
    end

    it 'logs feature importance as a bar plot' do
      expect(Wandb::Table).to receive(:new).with(data: [['feature1', 0.5], ['feature2', 0.3]], columns: %w[Feature Importance])
      expect(Wandb::Plot).to receive(:bar).with(mock_table, 'Feature', 'Importance', title: 'Feature Importance')
      expect(Wandb).to receive(:log).with({ 'Feature Importance' => mock_plot })

      callback.send(:log_feature_importance, mock_model)
    end
  end

  describe '#define_metric' do
    let(:callback) { described_class.new }

    it 'defines minimize metrics correctly' do
      expect(Wandb).to receive(:define_metric).with('train-error', summary: 'min')
      callback.send(:define_metric, 'train', 'error')
    end

    it 'defines maximize metrics correctly' do
      expect(Wandb).to receive(:define_metric).with('valid-auc', summary: 'max')
      callback.send(:define_metric, 'valid', 'auc')
    end

    it 'does not define metrics for unknown metric names' do
      expect(Wandb).not_to receive(:define_metric)
      callback.send(:define_metric, 'unknown', 'metric')
    end
  end
end

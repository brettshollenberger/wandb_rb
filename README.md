# Ruby Weights & Biases

A Ruby integration for the Weights & Biases AI platform. Log model and visualize model runs easily.

## Example Integration

```ruby
require "xgb"
require "wandb"
require "easy_ml"
require "polars-df"

df = Polars::DataFrame.new({
    "annual_revenue" => [1000, 2000, 3000],
    "loan_purpose" => %w[payroll expansion marketing],
    "rev" => [100, 200, 300],
    "date" => %w[2021-01-01 2021-05-01 2022-01-01],
}).with_column(
    Polars.col("date").str.strptime(Polars::Datetime, "%Y-%m-%d")
)

dataset = EasyML::Data::Dataset.new({
    datasource: df,
    target: "rev",
    splitter: { date: { date_col: "date", months_test: 2, months_valid: 2 } }
})

Wandb.login(api_key: "abc")
Wandb.init(project: "my-sweet-project")

model = EasyML::Core::Models::XGBoost.new(
  task: :regression,
  dataset: dataset,
  callbacks: [
    Wandb::XGBoostCallback.new(
        log_model: true,
        log_feature_importance: true,
        importance_type: :gain
    )
  ],
  hyperparameters: {
    learning_rate: 0.05,
    max_depth: 8,
    n_estimators: 150,
    booster: "gbtree",
    objective: "reg:squarederror"
  }
)

model.fit
wandb.finish
```

## Installation

```bash
gem install wandb
```

## Usage

Currently has deep integration with XGBoost Ruby, but if you want to manually log metrics with other tools:

```ruby
Wandb.login(api_key: "your-key")
Wanbd.init(project: "my-project")
Wandb.log({
    metric: 123,
    other_metric: 456
})
```

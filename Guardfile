guard :rspec, cmd: "bundle exec rspec" do
  watch(%r{^spec/.+\.rb$}) { |_m| "spec/xgboost_callback_spec.rb" }
  watch(%r{^lib/(.+)\.rb$}) { |_m| "spec/xgboost_callback_spec.rb" }
  watch(%r{^app/(.+)\.rb$}) { |_m| "spec/xgboost_callback_spec.rb" }
end

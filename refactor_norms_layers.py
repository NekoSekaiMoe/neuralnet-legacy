import re

with open('include/neuralnet/layer.h', 'r', encoding='utf-8') as f:
    content = f.read()

bn1d_regex = re.compile(r'    class BatchNorm1d : public Layer\s*\{.*?\n    };\n', re.DOTALL)
bn1d_new = """    class BatchNorm1d : public Layer
    {
    private:
        std::size_t num_features_;
        double eps_;
        double momentum_;
        bool affine_;

        Tensor gamma_;
        Tensor beta_;
        Matrix running_mean_;
        Matrix running_var_;
        int64_t num_batches_tracked_{0};
        bool is_training_{true};

    public:
        BatchNorm1d(std::size_t num_features, double eps = 1e-5, double momentum = 0.1, bool affine = true)
            : num_features_(num_features), eps_(eps), momentum_(momentum), affine_(affine),
              gamma_(affine ? Matrix(num_features, 1, 1.0) : Matrix(), affine),
              beta_(affine ? Matrix(num_features, 1, 0.0) : Matrix(), affine),
              running_mean_(num_features, 1, 0.0),
              running_var_(num_features, 1, 1.0)
        {
            if (num_features_ == 0) throw std::invalid_argument("BatchNorm1d: num_features > 0");
        }

        [[nodiscard]] const Matrix &running_mean() const noexcept { return running_mean_; }
        [[nodiscard]] const Matrix &running_var() const noexcept { return running_var_; }
        [[nodiscard]] int64_t num_batches_tracked() const noexcept { return num_batches_tracked_; }
        [[nodiscard]] bool is_training() const noexcept { return is_training_; }

        std::vector<Tensor> parameters() override
        {
            if (!affine_) return {};
            return {gamma_, beta_};
        }

        [[nodiscard]] std::size_t param_count() const noexcept override
        {
            return affine_ ? 2 * num_features_ : 0;
        }

        void on_mode_change(bool training) override { is_training_ = training; }

        const char *name() const override { return "BatchNorm1d"; }

        Tensor forward(const Tensor &input) override
        {
            return batchnorm1d(input, gamma_, beta_, running_mean_, running_var_, num_batches_tracked_, momentum_, eps_, is_training_, affine_);
        }
        
        void save_state(std::ostream &os) const override
        {
            auto write_mat = [&](const Matrix &m)
            {
                const std::size_t rows = m.rows();
                const std::size_t cols = m.cols();
                os.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
                os.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
                os.write(reinterpret_cast<const char *>(m.data().data()), m.size() * sizeof(double));
            };
            write_mat(running_mean_);
            write_mat(running_var_);
            os.write(reinterpret_cast<const char *>(&num_batches_tracked_), sizeof(num_batches_tracked_));
        }

        void load_state(std::istream &is) override
        {
            auto read_mat = [&](Matrix &m)
            {
                std::size_t rows = 0, cols = 0;
                is.read(reinterpret_cast<char *>(&rows), sizeof(rows));
                is.read(reinterpret_cast<char *>(&cols), sizeof(cols));
                if (rows != m.rows() || cols != m.cols()) throw std::runtime_error("BatchNorm1d: state shape mismatch");
                is.read(reinterpret_cast<char *>(m.data().data()), m.size() * sizeof(double));
            };
            read_mat(running_mean_);
            read_mat(running_var_);
            is.read(reinterpret_cast<char *>(&num_batches_tracked_), sizeof(num_batches_tracked_));
        }
    };\n"""
content = bn1d_regex.sub(bn1d_new, content)

bn2d_regex = re.compile(r'    class BatchNorm2d : public BatchNorm1d\s*\{.*?\n    };\n', re.DOTALL)
bn2d_new = """    class BatchNorm2d : public BatchNorm1d
    {
    public:
        using BatchNorm1d::BatchNorm1d;
        const char *name() const override { return "BatchNorm2d"; }
    };\n"""
content = bn2d_regex.sub(bn2d_new, content)

ln_regex = re.compile(r'    class LayerNorm final : public Layer\s*\{.*?\n    };\n', re.DOTALL)
ln_new = """    class LayerNorm final : public Layer
    {
    private:
        std::size_t normalized_shape_;
        double eps_;
        Tensor gamma_;
        Tensor beta_;

    public:
        explicit LayerNorm(std::size_t normalized_shape, double eps = 1e-5)
            : normalized_shape_(normalized_shape), eps_(eps),
              gamma_(Matrix(normalized_shape, 1, 1.0), true),
              beta_(Matrix(normalized_shape, 1, 0.0), true) {}

        const char *name() const override { return "LayerNorm"; }
        [[nodiscard]] std::size_t param_count() const noexcept override { return 2 * normalized_shape_; }

        std::vector<Tensor> parameters() override
        {
            return {gamma_, beta_};
        }

        Tensor forward(const Tensor &input) override
        {
            return layernorm(input, gamma_, beta_, eps_);
        }
    };\n"""
content = ln_regex.sub(ln_new, content)

with open('include/neuralnet/layer.h', 'w', encoding='utf-8') as f:
    f.write(content)

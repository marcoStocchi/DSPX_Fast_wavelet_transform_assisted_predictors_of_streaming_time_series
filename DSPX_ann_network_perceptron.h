// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once


namespace artificial_neural_networks
{

	template <class _Functype/*logistic, hyperbolic_tangent, etc.*/,
				class _InputLayerType /*instantiate with different input layer typenames*/,
					template <class> class... _LayerTypes /*instantiate with different hidden or output layer typenames*/>
	class general_multi_layer_perceptron /*undef*/; // general multilayer perceptron class


	template <class _Functype, 
		template <class> class... _LayerTypes>
	class general_multi_layer_perceptron <_Functype, input_layer, _LayerTypes...>
		: public network<input_layer, _LayerTypes<_Functype>...>
	{
		// specialize general MLP with standard raw input_layer copier,
		// retain parametric polymorphism for activation function, hidden and output layer typenames

		typedef network_type					base;

	public:

		typedef _Functype						function_type;
		typedef input_layer						input_layer_type;
		typedef perceptron_layer<_Functype>		perceptron_layer_type;
		typedef output_layer<_Functype>			output_layer_type;


		template <class... _Sizes>
		general_multi_layer_perceptron(_Sizes... _Sz)
			: base(_Sz...)
			, _training_pass(0)
		{
			_initialize();
		}

		~general_multi_layer_perceptron() {}

		
		auto input_size() const  -> size_t { return _input_layer().size(); }

		auto output_size() const  -> size_t { return _output_layer().size(); }


		void reinitialize() { _initialize(); }


		void set_learning_rate(const real_type& LR) {_RSset_learning_rate(LR);}

		auto get_learning_rate() const ->real_type { return _output_layer().get_learning_rate(); }


		template <class _inIt1, class _inIt2>
		void train_single(const _inIt1& _Beg, const _inIt1& _End, const _inIt2& _Act)
		{// train network to pattern _Beg->_End to the value in _Act

			_RSfeed(_Beg, _End);

			_RSbackpropagate(_Act);

			++_training_pass;
		}

		template <class _InIt>
		typename _InIt::value_type
			test_single(const _InIt& _Beg, const _InIt& _End)
		{// test single range

			_RSfeed(_Beg, _End);

			real_type val(0);

			_output_layer().output(&val);

			return val;
		}

		template <class _InIt, class _OutIt>
		void test_single(const _InIt& _Beg, const _InIt& _End, _OutIt _OBeg)
		{// test single input range, multiple outputs

			_RSfeed(_Beg, _End);
			
			_output_layer().output(_OBeg);
		}


		auto training_patterns() const ->size_t { return _training_pass; }

		void reset_training_patterns() { _training_pass=0; }

		bool load(const path_type& P)
		{
			return _load(P);
		}

		void save(const path_type& P) const {_save(P);}

		void dump(const path_type& P) const
		{
			_dump(P);
		}

	private:

		void _initialize()
		{
			_RSinitialize();
		}

		void _RSinitialize()
		{
			// start parametric recursion
			_initialize<1>();
		}

		template <size_t I>
		void _initialize()
		{
			_layer<I>().initialize(_layer<I-1>().size());

			_initialize<I+1>();
		}

		template <>
		void _initialize<tuple_size_type::value>() {/*stop recursion*/ }


		template <class _inIt>
		void _RSfeed(const _inIt& _Beg, const _inIt& _End)
		{
			_input_layer().feed(_Beg, _End); // raw feed

			// start parametric recursion
			_feed<1>();
		}

		template <size_t I>
		void _feed()
		{
			// feed next layer with previous' output
			_layer<I>().feed(_layer<I-1>());

			// recurr...
			_feed<I+1>();
		}

		template <>
		void _feed<tuple_size_type::value>() {/*stop recursion*/}

		
		template <class _inIt>
		void _RSbackpropagate(const _inIt& _Act)
		{
			// update output neurons weights, and prepare previous layer to backpropagation

			_output_layer().back_propagate(_Act, _layer<tuple_size_type::value-2>(), _training_pass);

			// start backward recursion
			_backpropagate<tuple_size_type::value-2>();
		}

		template <size_t I>
		void _backpropagate()
		{
			// use backpropagation deltas from the output layer/upper layers to update perceptrons weights;
			// in a multilayered network, perceptron layer deltas should be backpropagated to the prev hidden layer...
			_layer<I>().back_propagate(_layer<I-1>(), _training_pass);

			// recurr...
			_backpropagate<I-1>();
		}

		template <>
		void _backpropagate<0> () {/*stop recursion*/ }


		void _RSset_learning_rate(const real_type& LR)
		{
			// start backward recursion
			_set_learning_rate<tuple_size_type::value-1>(LR);
		}

		template <size_t I>
		void _set_learning_rate(const real_type& LR)
		{
			// ...
			_layer<I>().set_learning_rate(LR);

			// recurr...
			_set_learning_rate<I-1>(LR);
		}

		template <>
		void _set_learning_rate<0>(const real_type& LR) {/*stop recursion*/}



		size_t _training_pass;	// number of training steps done
	};
}


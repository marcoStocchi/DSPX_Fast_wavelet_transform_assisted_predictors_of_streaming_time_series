// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once
#define _USE_MATH_DEFINES
#include <math.h>

namespace artificial_neural_networks
{
	struct perceptron_base : public real_random_initializer_neuron
	{
		perceptron_base() : _lr(0.9), _prod(0) {}

		~perceptron_base() {}


		bool load(std::istream& s)
		{
			// vector resized
			s >> _bias(); s.ignore(1);

			for (auto I= _weights_begin(), E=_weights_end(); I!=E; ++I)
			{
				s >> *I; s.ignore(1);
			}

			return s.good();
		}

		void save(size_t _NeuronNo, std::ostream& s) const 
		{
			s << "Neuron " << _NeuronNo << " ";

			s << _prod << " " << output() << " | " << _bias() << " ";

			for (auto I= _weights_cbegin(), E=_weights_cend(); I!=E; ++I)
			{
				s << *I << " ";
			}

			s << "\n";
		}

		template <class _LayerType>
		void feed(const _LayerType& _L)
		{
			// feed coming from a type of perceptron layer
			_inputV.clear(); _inputV.resize(_L.size());

			std::transform(_L.cbegin(), _L.cend(), _inputV.begin(),
				[](const _LayerType::neuron_type& _N) {return _N.output(); });

			_prod = _bias() + dot_product(_inputV.cbegin(), _inputV.cend(), _weights_cbegin());
		}

		template <>
		void feed <input_layer>(const input_layer& _L)
		{
			// feed coming from a raw-input layer
			_inputV.clear(); _inputV.resize(_L.size());
			
			std::copy(_L.cbegin(), _L.cend(), _inputV.begin());

			_prod = _bias() + dot_product(_inputV.cbegin(), _inputV.cend(), _weights_cbegin());
		}

		real_type _learning_rate(const size_t& trainings) const { return _lr; } 


		void set_learning_rate(const real_type& LR) { _lr=LR; }
	
		auto get_learning_rate() const ->real_type { return _lr; }

	protected:

		void _min_learning_rate_test() { if (_lr<0.01) _lr=0.01; }

		void _max_learning_rate_test() { if (_lr>0.9) _lr=0.9; }

		void _initialize(const real_type& _m, const real_type& _M, const size_t& sz)
		{
			// create weights according to the input layer size
			// and to the random initializer method - see neuron.h	

			weights_initialize(sz, _m, _M);

			bias_initialize(_m, _M);
		}

		real_type				_lr;
		real_type				_prod;		// the dot product, calculated on input feed
		real_vector_type		_inputV;	// store input values temporarily
	};

	template <class _ActivFunc>
	struct weight_initializator /*undef*/;

	template <>
	struct weight_initializator <logistic> : public perceptron_base
	{
		void initialize(const size_t& _Sz) { perceptron_base::_initialize(0.1, 0.9, _Sz); }
	};

	template <>
	struct weight_initializator <hyperbolic_tangent> : public perceptron_base
	{
		void initialize(const size_t& _Sz) { perceptron_base::_initialize(-0.9, 0.9, _Sz); }
	};

	template <class _ActivFunc>
	class perceptron : public weight_initializator <_ActivFunc>
	{// Rosenblatt, Ruhmelhart et alt.
		// sigmoid activation, backpropagation
		

	public:
			
		perceptron() {}

		perceptron(const perceptron&) {}

		~perceptron() {}


		auto output() const->real_type
		{
			// activation func call
			return _ActivFunc::execute(_prod);
		}

		auto output_delta(const real_type& _Outp, const real_type& _SumDeltas) const ->real_type
		{
			real_type _der = _ActivFunc::derivative(_Outp);

			// find delta
			real_type _Delta(_SumDeltas * _der);

			return _Delta;
		}

		void push_back_propagate(const real_type& _UpperDelta)
		{
			// push upper layer deltas
			_deltas.push_back(_UpperDelta);
		}

		template <class _LayerType>
		void back_propagate(_LayerType& _L, const size_t& trainings)
		{
			// sum accumulated deltas from upper layer
			real_type _SUMdelta(sum(_deltas.begin(), _deltas.end()));

			// calculate sigmoid of the product
			real_type _Outp(output());

			// execute derivative of sigmoid
			real_type deltaI = output_delta(_Outp, _SUMdelta);

			// learning rate ?
			real_type LR = _learning_rate(trainings); 

			// update weights
			for (auto WI= _weights_begin(), XI = _inputV.begin(), WE=_weights_end(); WI!=WE; ++WI, ++XI)
			{
				// Multiply its output delta and input activation to get the gradient of the weight.
				// Subtract a ratio(percentage) of the gradient from the weight.

				*WI = *WI + LR* ((*XI) * deltaI);
			}

			_bias() += LR * deltaI;

			// back propagate to prev layer...
			// for each previous layer perceptron, push the product
			// delta * new_respective_weight
			auto WI= _weights_begin();

			for (auto HI=_L.begin(), HE=_L.end(); HI!=HE; ++HI, ++WI)
			{// iterate prev layer neurons directly to give each of them their respective value
				HI->push_back_propagate(*WI * deltaI);
			}

			// remove deltas values to get ready for the next training step
			_deltas.clear();
		}

		template <>
		void back_propagate <input_layer> (input_layer& _L, const size_t& trainings)
		{
			// sum accumulated deltas from upper layer
			real_type _SUMdelta(sum(_deltas.begin(), _deltas.end()));

			// calculate sigmoid of the product
			real_type _Outp(output());

			// execute derivative of sigmoid
			real_type deltaI = output_delta(_Outp, _SUMdelta);

			// learning rate ?
			real_type LR = _learning_rate(trainings);

			// update weights
			for (auto WI= _weights_begin(), XI = _inputV.begin(), WE=_weights_end(); WI!=WE; ++WI, ++XI)
			{
				// Multiply its output delta and input activation to get the gradient of the weight.
				// Subtract a ratio(percentage) of the gradient from the weight.

				*WI = *WI + LR*((*XI) * deltaI);
			}

			_bias() += LR * deltaI;


			// remove deltas values to get ready for the next training step
			_deltas.clear();
		}

	private:

		real_vector_type		_deltas;	// upper layer backpropagation factors
	};
}



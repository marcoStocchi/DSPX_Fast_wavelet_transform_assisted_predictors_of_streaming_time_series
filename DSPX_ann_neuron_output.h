// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once
#undef max

namespace artificial_neural_networks
{

	template <class _ActivFunc>
	class output_neuron : public perceptron_base
	{// output neurons

	public:

		output_neuron()	{}

		~output_neuron() {}


		void initialize(size_t _Sz) { perceptron_base::_initialize(0.1, 0.9, _Sz); }

		virtual auto output() const->real_type	{ return _ActivFunc::execute(_prod); }

		virtual auto output_delta(const real_type& _Outp, const real_type& _Error) ->real_type
		{
			real_type _der = _ActivFunc::derivative(_Outp);

			// find delta
			real_type _Delta(_Error * _der);

			return _Delta;
		}

		template <class _LayerType>
		void back_propagate(const real_type& _Desired, _LayerType& _L, const size_t& trainings, bool _Bdebug=false)
		{
			// calculate sigmoid of the product
			real_type _Outp(output());

			// error for this output neuron
			real_type _Error =(_Desired - _Outp);

			real_type deltaI(output_delta(_Outp, _Error));


			// learning rate ?
			real_type LR = _learning_rate(trainings);

			for (auto WI= _weights_begin(), XI = _inputV.begin(), WE=_weights_end(); WI!=WE; ++WI, ++XI)
			{
				*WI = *WI + LR* ((*XI) * deltaI);
			}

			_bias() += LR * deltaI; //_Error;

			// for each previous layer perceptron, push the product
			// delta * new_respective_weight
			auto WI= _weights_begin();

			for (auto HI=_L.begin(), HE=_L.end(); HI!=HE; ++HI, ++WI)
			{// iterate prev layer neurons directly to give each of them their respective value
				HI->push_back_propagate(*WI * deltaI);
			}
		}
	};

	
}
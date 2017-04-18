// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once


namespace artificial_neural_networks
{
	template <class _ActivFunc>
	class output_layer : public layer<std::vector<output_neuron<_ActivFunc>>>
	{
		// output class for predictor networks

		typedef output_neuron<_ActivFunc>				perceptron_type;
		typedef layer<std::vector<perceptron_type>>		base;

	public:

		typedef perceptron_layer<_ActivFunc>			hidden_layer_type;


		output_layer(const size_t& _Sz)
			: base(_Sz)
		{}

		~output_layer(){}


		void initialize(const size_t& _InputSz)
		{
			// iterate initializator functor bound to layer size
			base::_initialize(_InputSz);
		}

		void feed(const hidden_layer_type& _L)
		{
			// feed forward each neuron of the _L layer
			// using general purpose protected iteration function _foreach()
			_foreach(std::bind2nd(std::mem_fun_ref(&perceptron_type::feed<hidden_layer_type>), _L));
		}

		template <class _InIt>
		void back_propagate(const _InIt& _DesiredBeg, hidden_layer_type& _L, const size_t& trainings)
		{
			// the number of output neurons is = to the size of _Desired range

			auto DESIRED(_DesiredBeg); // first desired value iterator!

			for (auto I= begin(), E= end(); I!=E; ++I, ++DESIRED)
			{// iterate through the output neurons
				// each output neuron has a weight to the ith perceptron of the previous layer

				// each output neuron must calculate its delta, update its weights,
				// and backpropagate the quantity delta*new_weight to the perceptron the new_weight links to.
				
				// push_back propagation to layer's _L neurons
				I->back_propagate(*DESIRED, _L, trainings); 
			}
		}

		template <class OutIt>
		void output(OutIt _Beg)
		{
			auto DEST(_Beg);

			for (auto I= begin(), E= end(); I!=E; ++I)

				*DEST++ = I->output();
		}

	};
}
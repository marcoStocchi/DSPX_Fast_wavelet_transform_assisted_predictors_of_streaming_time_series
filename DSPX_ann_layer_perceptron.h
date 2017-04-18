// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once


namespace artificial_neural_networks
{
	template <class _Functype> // eg. logistic, hyberbolic_tangent...
	class perceptron_layer : public layer<std::vector<perceptron<_Functype>>>
	{
		typedef layer<std::vector<perceptron<_Functype>>>	base;

	public:

		typedef base										perceptron_layer_type;
		typedef perceptron<_Functype>						perceptron_type;


		perceptron_layer(const size_t& _Sz) : base(_Sz) {}

		~perceptron_layer() {}


		void initialize(const size_t& _InputSz)
		{
			// iterate initializator functor bound to layer size
			base::_initialize(_InputSz);
		}

		template <class _LayerType>
		void feed(const _LayerType& _L)
		{
			// feed forward each neuron of the _L layer	
			_foreach(std::bind2nd(std::mem_fun_ref(&perceptron_type::feed<_LayerType>), _L));
		}

		template <class _LayerType>
		void back_propagate(_LayerType& _L, const size_t& trainings)
		{
			for (auto I= begin(), E= end(); I!=E; ++I)
			{// iterate through the prev hidden neurons
				// each neuron of this layer has a weight to the ith perceptron of the previous layer

				// each neuron of this layer must calculate its delta, update its weights,
				// and backpropagate the quantity delta*new_weight to the perceptron the new_weight links to.
				I->back_propagate(_L, trainings);
			}
		}
	};
}
// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

namespace artificial_neural_networks
{

	class input_layer : public layer<std::vector<real_type>>
	{
		// provides conventional raw input copy into the network

		typedef layer<std::vector<real_type>>		base;

	public:


		input_layer(const size_t& _Sz)
			: base(_Sz)
		{}

		~input_layer() {}


		template <class _inIt>
		void feed(const _inIt& _Beg, const _inIt& _End)
		{
			std::copy(_Beg, _End, begin());
		}


		void dump_inputs(std::ostream& s, char _Endl='\n') const
		{
			for (auto I=cbegin(), E=cend(); I!=E; ++I)

				s << *I << " ";

			s << _Endl;
		}

	private:
	};
}
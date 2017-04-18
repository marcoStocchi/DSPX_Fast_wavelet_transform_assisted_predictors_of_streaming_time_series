// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once


namespace artificial_neural_networks
{
	typedef double											real_type;

	typedef std::default_random_engine						random_engine_type;
	typedef std::uniform_real_distribution<real_type>		random_unif_type;

	typedef std::vector<real_type>							real_vector_type;
	typedef real_vector_type::const_iterator				real_vector_const_iterator;
	typedef real_vector_type::iterator						real_vector_iterator;

	typedef std::vector<real_vector_type>					real_matrix_type;
	typedef real_matrix_type::const_iterator				real_matrix_const_iterator;		// rows iterator
	typedef real_matrix_type::iterator						real_matrix_iterator;			// rows iterator


	extern random_engine_type	_Re;

	extern const real_type		_Epsilon; // ~ e-16* per macchina x64
	
	extern const real_type		_Infinity;
}

namespace ann = artificial_neural_networks;

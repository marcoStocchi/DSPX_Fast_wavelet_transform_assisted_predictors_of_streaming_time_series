// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

namespace artificial_neural_networks
{
	// help fnctions to test and train neural networks...


	template <class _NetworkType> inline 
		real_type network_test_single(_NetworkType& _Net, const real_vector_type& _In)
	{
		return _Net.test_single(_In.cbegin(), _In.cend());
	}


	template <class _NetworkType> inline 
		void network_train_single(
				_NetworkType& _Net, const real_vector_type& _In,
					real_type _Err, const real_type& _MaxErr, const real_type& _MinErr,
						const real_type& _sdActual)
	{
		if (std::abs(_Err) > _MaxErr)
		{
			//cout << _Err << " " << _MaxErr << "\n";		// uncomment for debug

			while (std::abs(_Err) > _MinErr)
			{
				_Net.train_single(_In.cbegin(), _In.cend(), &_sdActual);

				real_type _sdNew = _Net.test_single(_In.cbegin(), _In.cend());

				_Err = _sdNew - _sdActual;
			}
		}

		//else cout << "no substantial error\n";			// uncomment for debug
	}

}
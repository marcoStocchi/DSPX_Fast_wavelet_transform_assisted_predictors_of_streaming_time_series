// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

namespace artificial_neural_networks
{
	// random

	extern random_engine_type _Re;

	struct _RealDraw
	{// random engine wrapper drawing from a uniform distribution

		_RealDraw(const real_type& _m, const real_type& _M) : _Rd(_m, _M) {}
		
		~_RealDraw() {}


		real_type operator() () const { return _Rd(_Re); }

	private:

		random_unif_type	_Rd;
	};

	template <class C> inline
		void _UninitializedRand(C& _Cont, const size_t& _Size, const real_type& _m, const real_type& _M)
	{
		_Cont.resize(_Size);

		std::generate(_Cont.begin(), _Cont.end(), _RealDraw(_m, _M));
	}

	inline void _UninitializedRand(real_type& _Val, const real_type& _m, const real_type& _M)
	{
		_Val= _RealDraw(_m, _M)();
	}

	inline real_type _UninitializedRand(const real_type& _m, const real_type& _M)
	{
		return _RealDraw(_m, _M)();
	}


	// math ops, statistics

	template <class inIt> inline size_t
		count(const inIt& _Beg, const inIt& _End)
	{
		return std::distance(_Beg, _End);
	}

	template <class inIt> inline typename inIt::value_type
		sum(const inIt& _Beg, const inIt& _End)
	{
		typename inIt::value_type r(0);

		for (auto I=_Beg, E=_End; I!=E; ++I) r+=*I;

		return r;
	}

	template <class inIt> inline typename inIt::value_type
		mean(const inIt& _Beg, const inIt& _End)
	{
		const size_t _Cnt(count(_Beg, _End));

		if (!_Cnt) return (typename inIt::value_type) 0;

		return sum(_Beg, _End) / _Cnt;
	}

	template <class C> inline typename C::value_type
		mean(const C& _Cont)
	{
		return mean(_Cont.cbegin(), _Cont.cend());
	}

	template <class inIt> inline typename inIt::value_type
		dot_product(const inIt& _Beg1, const inIt& _End1, const inIt& _Beg2)
	{// fastest dot prod implementation
		typename inIt::value_type r(0);

		for (auto I1=_Beg1, I2=_Beg2, E=_End1; 
				I1!=E; ++I1, ++I2)
					r+= (*I1) * (*I2);

		return r;
	}


	// functors

	struct logistic
	{// sigmoid 0;1
		static const real_type _Lambda1;

		static real_type execute(const real_type& x) { return 1/ (1+ std::pow(M_E, -x/_Lambda1)); }

		static real_type derivative(const real_type& x) { return x * (1-x); }

		static real_type invert(const real_type& x) { return -_Lambda1 * std::log(1.0/x - 1.0); }
	};

	const real_type logistic::_Lambda1 = 2.0;


	struct hyperbolic_tangent
	{// sigmoid -1;1
		static const real_type _Lambda2;

		static real_type execute(const real_type& x) { return std::tanh(x/_Lambda2); } //{ return std::tanh(0.5*x); }

		static real_type derivative(const real_type& x) { return 1- std::pow(x,2); }

		static real_type invert(const real_type& x) { return _Lambda2 * std::log((1.0+x)/(1.0-x))/2; }
	};

	const real_type hyperbolic_tangent::_Lambda2 = 80.0;
}

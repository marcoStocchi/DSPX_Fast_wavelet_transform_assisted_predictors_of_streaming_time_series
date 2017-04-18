// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// implementation of fast wavelet transform
// Marco Stocchi - UNICA
// version 1.2

// log 
// v 1.1: implementation of shift variance theorem for streaming datasets
// v 1.2: refactoring, commentwork

#pragma once


namespace fast_wavelet_transform
{
	struct shift_variance_theorem
	{
		size_t						_Srcsize;	// time series size
		size_t						_N;			// wavelet support
		size_t						_maxjexp;	// max scaling exponent
		size_t						_scalingsz;	// number of scaling coefficients
		std::map<size_t, size_t>	_varc;		// map j -> variant coefficients
		std::map<size_t, size_t>	_bands;		// map j -> indeces

		shift_variance_theorem(const size_t& Srcsize, const size_t& _WaveletN)
			: _Srcsize(Srcsize)
			, _N(_WaveletN)
			, _maxjexp(_maxj())
			, _scalingsz(_scaling_size())
		{
			_fill_variant_coefficients();

			_fill_subbands();
		}

		~shift_variance_theorem()
		{}

		bool is_scaling_coefficient(const size_t& i) const {return i<_scalingsz;}

		bool is_SVT_coefficient(const size_t& i) const
		{
			// bool if a coefficient can be retrieved from previous wavelet transforms

			try 
			{
				const size_t j(_index_to_scale(i));

				return i < _bands.at(j - 1) - _variant_coefficients_size(j);
			}

			catch (...) {return false;}
		}

		auto back_steps(const size_t& i) const ->size_t
		{
			// number of history backsteps required in order to...
			// ... retrieve the coefficient in the rightish column
			// ... of matrix Q
			return (size_t)(std::pow(2, _index_to_scale(i)));
		}

		auto maxj() const ->size_t {return _maxjexp;}

		auto source_size() const ->size_t {return _Srcsize;}

		auto scaling_size() const ->size_t {return _scalingsz;}

		auto wavelet_support() const ->size_t {return _N;}

		void variant_coefficients(std::vector<size_t>& _VarCoeff) const
		{
			// extract the number of non-SVT coefficients... 
			// see paper DSPX (table 1)

			_VarCoeff.clear();

			for (auto I= _varc.cbegin(), E= _varc.cend(); I!=E; ++I)
				_VarCoeff.push_back(I->second);
		}

	private:

		auto _variant_coefficients_size(const size_t& j) const ->size_t
		{
			// return the number of shift-variant coefficients at scale 'j'
			return _varc.at(j);
		}

		void _fill_subbands()
		{
			// fill the size of subbands
			// see paper DSPX (eq. 12)

			for (size_t j = 0; j <= _maxjexp; ++j)
			{
				_bands[j] = (_Srcsize >> j);
			}
		}

		void _fill_variant_coefficients()
		{
			// see paper DSPX (eq. system 13)
			// Lemma on the number of variant coefficients

			_varc[1] = _N; // see Nh/2

			for (size_t i = 2; i <= _maxjexp; ++i)

				_varc[i] = (_varc[i - 1] + 1) / 2 + (_varc[i - 1] + 1) % 2 + _N - 1;
		}

		auto _index_to_scale /*throws*/(const size_t& i) const ->size_t
		{// throws if 'i' is a scaling coefficient
			for (auto I = _bands.cbegin(), E = _bands.cend(); I != E; ++I)

				if (I->second <= i) return I->first;

			throw std::exception("DWT theorem: _index_to_scale - failed, index out of range");
		}

		auto _scaling_size() const ->size_t 
		{// returns number of scaling coefficients, depending on the size of the wavelet support

			return (size_t) (std::pow(2, std::ceil(std::log2(2*_N))));
		}

		auto _maxj() const ->size_t
		{// returns max DWT depth, depending on the size of the source and the size of the wavelet support

			return (size_t) (std::log2(_Srcsize) - std::ceil(std::log2(2*_N)));
		}
	};

	struct DWT
	{
		typedef double									floating_point_type;
		typedef floating_point_type						value_type;
		typedef value_type*								pointer;
		typedef const value_type*						const_pointer;

		typedef std::vector<value_type>					vector_type;
		typedef std::vector<vector_type>				matrix_type;


		virtual ~DWT() {}

		virtual auto size() const->size_t =0;

		virtual void transform(const const_pointer&, 
								const pointer&, const size_t&) const =0;

		virtual void transform(const std::vector<size_t>& _VariantCoeff /*cached variant sizes*/,
				const std::vector<size_t>& _Backsteps /*cached backsteps for theorem transposition*/,
					const matrix_type& _Q /*const reference to matrix Q*/,
						const const_pointer& _Src /*ptr to source series*/, 
							const pointer& _Dest /*destination ptr to wavelet transform*/, 
								const size_t& _N /*source size*/) const =0;

		virtual void invert(const const_pointer&, 
							const pointer&, const size_t&) const =0;

		virtual void coefficients(std::vector<value_type>&) const =0;

		virtual auto wavelet_type() const ->std::string =0;
	};

	template <size_t _FilterN>
	struct DWT_base : DWT
	{
		typedef std::array<floating_point_type, 2*_FilterN>		array_type;

	protected:

		template <class... _Coeff>
		DWT_base(_Coeff... c)
			: _CacheSz(2*_FilterN)
			, _CacheBaseSz(static_cast<const size_t>(std::pow(2.0, std::ceil(std::log2(_CacheSz)))))
			, _H({ c... }) 
			, _G(_fillg())
			, _Ih(_invertH())
			, _Ig(_invertG())
		{
		}

		~DWT_base() {}


	public:

		virtual auto size() const ->size_t { return _CacheSz; }


		virtual void transform(const const_pointer& _Src, 
				const pointer& _Dest, 
					const size_t& _N) const
		{
			// Fast Wavelet Transform
			// N must be a power of 2 
			
			if (_N<_CacheSz) throw std::exception("DWT failure, small range");

			_transform(_Src, _Dest, _N); // e.g 128

			for (size_t n=(_N>>1); n>=_CacheSz ; n>>=1) { // e.g. 64, 32, 16, 8, 4

				_transform(_Dest, _Dest /*overwrite*/, n);

				// 'dest' is transformed, only the first n coefficients
				// are transformed. The rightish coefficients remain 
				// those transformed the previous cycle.
			}
		}


		virtual void transform(const std::vector<size_t>& _VariantCoeff /*cached variant sizes*/,
				const std::vector<size_t>& _Backsteps /*cached backsteps for theorem transposition*/,
					const matrix_type& _Q /*const reference to matrix Q*/,
						const const_pointer& _Src /*ptr to source series*/, 
							const pointer& _Dest /*destination ptr to wavelet transform*/, 
								const size_t& _N /*source size*/) const
		{
			// theorem Discrete Wavelet Transform of dyadic series
			// N must be a power of 2 
			
			if (_N<_CacheSz ) throw std::exception("DWT failure, small range");


			const size_t _history_size(_Q.size());	// no. of rows of matrix Q

			size_t _Half(_N >> 1);	// cache 

			size_t j(0);	// iteration index

			const size_t* _VarCoeffptr(&_VariantCoeff[0]), 
				*_Backsteps_ptr(&_Backsteps[0]); // readonly ptrs 
		
			size_t _Imax(_N - *_VarCoeffptr - _Half);	// max iteration for theorem copy operations

			vector_type::const_pointer	// readony Q ptr
				_Qptr(&_Q[_history_size - *_Backsteps_ptr -1][_Half+1]);	

			vector_type::pointer	// write ptrs to destination DWT
				_Details_ptr(&_Dest[_Half]), 
					_Difference_ptr(&_Dest[0]);	


			// theorem copy...
			for (size_t i=0; i<_Imax; 
					++i, ++_Difference_ptr, ++++j)
			{
				// fast copy theorem coefficients...
				*_Details_ptr++ = *_Qptr++;		

				for (size_t z=0; z < _CacheSz ; ++z)
				
					*_Difference_ptr += _Src[j + z] * _H[z]; // convolve source and scaling			 
			}
			
			// non theorem transform steps (final for the first resolution step)...
			size_t _VariantSteps(_N/2-_Imax);

			for (size_t y=0; y<_VariantSteps; 
					++y, ++_Details_ptr, ++_Difference_ptr, ++++j)

				for (size_t z=0; z<_CacheSz; ++z)
				{
					*_Difference_ptr += _Src[(j+z)%_N] * _H[z]; // convolve source and scaling
						
					*_Details_ptr += _Src[(j+z)%_N] * _G[z]; // convolve source and wavelet
				}


			// start deeper scales transformation... >>>>>

			++_VarCoeffptr; ++_Backsteps_ptr; // next variant coeff size and backstep

			size_t n(_N>>1);	// cache

			std::vector<value_type> _Tmp(n); // temporary depot vector

			for (; n>_CacheBaseSz; n>>=1, 
				++_VarCoeffptr, ++_Backsteps_ptr) 
			{// e.g. 64, 32, 16, 8, 4

				_Half = n >> 1;

				_Tmp.assign(n, 0);

				vector_type::const_pointer	// read only
					_Qptr(&_Q[_history_size - *_Backsteps_ptr -1][_Half+1]);	
	
				_Details_ptr=&_Tmp[_Half]; _Difference_ptr=&_Tmp[0];

				_Imax=n-(*_VarCoeffptr)-_Half;

				j=0;

				for (size_t i=0; i<_Imax; 
						++i, ++_Difference_ptr, ++++j)
				{
					// fast copy coroll coeff...
					*_Details_ptr++ = *_Qptr++;		

					for (size_t z=0; z<_CacheSz ; ++z)

						*_Difference_ptr += _Dest[j + z] * _H[z];
				}

				_VariantSteps=n/2-_Imax;

				for (size_t y=0; y<_VariantSteps; 
						++y, ++_Details_ptr, ++_Difference_ptr, ++++j)

					for (size_t z=0; z<_CacheSz ; ++z)
					{
						*_Difference_ptr += _Dest[(j + z) % n] * _H[z]; // convolve...

						*_Details_ptr += _Dest[(j + z) % n] * _G[z]; // convolve...
					}

				// partial copy to effective destination...
				std::copy(_Tmp.cbegin(), n+_Tmp.cbegin(), _Dest);
			}


			// phi coefficients of the wavelet series... >>>>>>

			_Half = n >> 1;	// cache

			_Tmp.assign(n, 0);
	
			_Imax=n/2;

			_Details_ptr=&_Tmp[_Half]; _Difference_ptr=&_Tmp[0];

			j=0;

			for (size_t y=0; y<_Imax; 
				++y, ++_Details_ptr, ++_Difference_ptr, ++++j)

				for (size_t z=0; z<_CacheSz ; ++z)
				{
					*_Difference_ptr += _Dest[(j + z) % n] * _H[z]; // convolve...

					*_Details_ptr += _Dest[(j + z) % n] * _G[z]; // convolve...
				}


			// final transfer to effective destination vector...
			std::copy(_Tmp.cbegin(), n+_Tmp.cbegin(), _Dest);
		}

		void invert(const const_pointer& _Src, const pointer& _Dest, 
						const size_t& _N) const
		{
			if (_N<_CacheSz) throw std::exception("DWT failure, small range");

			std::copy(_Src, _Src+_N, _Dest);

			for (size_t n=_CacheBaseSz; n<=_N; n<<=1)
			{// 4, 8, 16, ...128
				_invTransform(_Dest, n);
			}
		}

		void coefficients(std::vector<floating_point_type>& _Out) const
		{
			// export wavelet coefficients in _Out
			_Out.clear();

			_Out.insert(_Out.end(), _H.cbegin(), _H.cend());
			_Out.insert(_Out.end(), _G.cbegin(), _G.cend());
			_Out.insert(_Out.end(), _Ih.cbegin(), _Ih.cend());
			_Out.insert(_Out.end(), _Ig.cbegin(), _Ig.cend());
		}

	private:

		void _transform(const const_pointer& _Src, const pointer& _Dest, 
							const size_t& _N) const
		{
			const size_t _Rs(_CacheSz /2 -1); // remaining steps

			const size_t _Half(_N >> 1);

			std::vector<value_type> _Tmp(_N);

			size_t i(0), j(0);

			for (; j<=_N-_CacheSz +1; ++i, ++++j)
			{
				for (size_t z=0; z<_CacheSz ; ++z)
				{
					_Tmp[i]      += _Src[j+z]*_H[z]; // convolve source and scaling

					_Tmp[i+_Half] += _Src[j+z]*_G[z]; // convolve source and wavelet
				}
			}

			for (size_t y=_Rs; y>0; --y, ++i, ++++j)
			{// trepassing series borders
				for (size_t z=0; z<_CacheSz ; ++z)
				{
					_Tmp[i]      += _Src[(j+z)%_N]*_H[z]; // convolve...

					_Tmp[i+_Half] += _Src[(j+z)%_N]*_G[z]; // convolve...
				}
			}

			std::copy(_Tmp.cbegin(), _Tmp.cend(), _Dest);
		}

		void _invTransform(const pointer& _Dest, const size_t& _N) const
		{
			const size_t _Is(_CacheSz/2 -1); // initial steps

			const size_t _Half(_N>>1);

			std::vector<value_type> _Tmp(_N);

			size_t j(0);

			for (size_t y=_Is; y>0; --y, ++++j)
			{// e.g. y= 3,2,1

				for (size_t z=0; z<_FilterN /*_CacheSz/2*/; ++z)
				{// e.g. z=0,1,2,3
					
					const size_t _left((_Half-y+z)%_Half);
				
					const size_t _right(((_Half-y+z)%_Half)+_Half);

					_Tmp[j] += _Dest[_left]*_Ih[2*z] + _Dest[_right]*_Ih[2*z+1];
				
					_Tmp[j+1] += _Dest[_left]*_Ig[2*z] + _Dest[_right]*_Ig[2*z+1];
				}
			}
			
			for (size_t i=0; i<_Half-_Is; ++i, ++++j)
			{
				for (size_t z=0; z<_FilterN /*_CacheSz/2*/; ++z)
				{// e.g. z=0,1,2,3
					_Tmp[j] += _Dest[i+z]*_Ih[2*z] + _Dest[i+_Half+z]*_Ih[2*z+1];
				
					_Tmp[j+1] += _Dest[i+z]*_Ig[2*z] + _Dest[i+_Half+z]*_Ig[2*z+1];
				}
			}

			std::copy(_Tmp.cbegin(), _Tmp.cend(), _Dest);
		}


		array_type _fillg() const 
		{
			// Daubechies, "Orthonormal bases of compactly supported wavelets"
			// equations 3.17, 3.45

			array_type _D;

			for (size_t i=0; i<_CacheSz ; ++i) 
				
				_D[i] = std::pow(-1, i) * _H[_CacheSz -i-1];

			return _D;
		}

		array_type _invertH() const
		{// create inverted H based on h and g

			// Retrieve the first index which covers all 
			// the necessary forward passages to be able 
			// to perform an inverted passage 
			// (see matrix at pos.2)

			const size_t istart = 2* (_CacheSz /2 -1);

			array_type _D;

			for (size_t i=0, z=istart; i<_CacheSz ; i+=2, z-=2)
			{	// difficult to comment in words, check anonymous 
				//namespace, matrix at col.2 (vertical iteration)
				_D[i]=_H[z];

				_D[i+1]=_G[z];
			}

			return _D;
		}

		array_type _invertG() const
		{// create inverted H based on h and g

			// same as _invertH()
			const size_t istart = 2* (_CacheSz /2 -1);

			array_type _D;

			for (size_t i=0, z=istart; i<_CacheSz ; i+=2, z-=2)
			{	// same as invertH(), except that we take elements 
				// from the next column of the matrix
				_D[i]=_H[z+1];

				_D[i+1]=_G[z+1];
			}

			return _D;
		}


		const size_t		_CacheSz;
		const size_t		_CacheBaseSz;	// next pwr of 2
		const array_type	_H;
		const array_type	_G;
		const array_type	_Ih;
		const array_type	_Ig;
	};

	template <size_t N>
	struct Daubechies_commnon : DWT_base<N>
	{
		template <class... _Coeff>
		Daubechies_commnon(_Coeff... c) 
			: DWT_base(c...) 
		{
			_Name << "Daubechies " << N;
		}

		~Daubechies_commnon() {}


		virtual auto wavelet_type() const ->std::string {return _Name.str();}

	private:

		std::ostringstream _Name;
	};

	template <size_t N>
	struct Daubechies /*undef*/;
	
	template <>
	struct Daubechies <2> : Daubechies_commnon<2>
	{
		Daubechies()
			: Daubechies_commnon(	0.48296291314469025, 0.83651630373746899, 
						0.22414386804185735, -0.12940952255092145)
		{}
	};

	template <>
	struct Daubechies <3> : Daubechies_commnon<3>
	{
		Daubechies()
			: Daubechies_commnon(	0.33267055295095688, 0.80689150931333875, 
						0.45987750211933132, -0.13501102001039084, 
						-0.085441273882241486, 0.035226291882100656)
		{}
	};

	template <>
	struct Daubechies <4> : Daubechies_commnon<4>
	{
		Daubechies()
			: Daubechies_commnon(	0.23037781330885523, 0.71484657055254153, 
						0.63088076792959036, -0.027983769416983849, 
						-0.18703481171888114, 0.030841381835986965,
						0.032883011666982945, -0.010597401784997278)
		{}
	};

	template <>
	struct Daubechies <5> : Daubechies_commnon<5>
	{
		Daubechies()
			: Daubechies_commnon(	0.16010239797412501, 0.60382926979747287, 
						0.72430852843857441, 0.13842814590110342, 
						-0.24229488706619015, -0.03224486958502952,
						0.077571493840065148, -0.0062414902130117052, 
						-0.012580751999015526, 0.0033357252850015492)
		{}
	};

	template <>
	struct Daubechies <6> : Daubechies_commnon<6>
	{
		Daubechies()
			: Daubechies_commnon(0.11154074335008017, 0.49462389039838539,
						0.75113390802157753, 0.3152503517092432,
						-0.22626469396516913, -0.12976686756709563,
						0.097501605587079362, 0.027522865530016288,
						-0.031582039318031156, 0.0005538422009938016,
						0.0047772575110106514, -0.0010773010849955799)
		{}
	};

	template <>
	struct Daubechies <7> : Daubechies_commnon<7>
	{
		Daubechies()
			: Daubechies_commnon(0.077852054085062364, 0.39653931948230575,
						0.72913209084655506, 0.4697822874053586,
						-0.14390600392910627, -0.22403618499416572,
						0.071309219267050042, 0.080612609151065898,
						-0.038029936935034633, -0.01657454163101562,
						0.012550998556013784, 0.00042957797300470274,
						-0.0018016407039998328, 0.00035371380000103988)
		{}
	};

	template <>
	struct Daubechies <8> : Daubechies_commnon<8>
	{
		Daubechies()
			: Daubechies_commnon(0.054415842243081609, 0.31287159091446592,
						0.67563073629801285, 0.58535468365486909,
						-0.015829105256023893, -0.28401554296242809,
						0.00047248457399797254, 0.12874742662018601,
						-0.017369301002022108, -0.044088253931064719,
						0.013981027917015516, 0.0087460940470156547,
						-0.0048703529930106603, -0.00039174037299597711,
						0.00067544940599855677, -0.00011747678400228192)
		{}
	};

	template <>
	struct Daubechies <9> : Daubechies_commnon<9>
	{
		Daubechies()
			: Daubechies_commnon(0.038077947363167282, 0.24383467463766728,
						0.6048231236767786, 0.65728807803663891,
						0.13319738582208895, -0.29327378327258685,
						-0.096840783220879037, 0.14854074933476008,
						0.030725681478322865, -0.067632829059523988,
						0.00025094711499193845, 0.022361662123515244,
						-0.004723204757894831, -0.0042815036819047227,
						0.0018476468829611268, 0.00023038576399541288,
						-0.00025196318899817888, 3.9347319995026124e-05)
		{}
	};

	template <>
	struct Daubechies <10> : Daubechies_commnon<10>
	{
		Daubechies()
			: Daubechies_commnon(0.026670057900950818, 0.18817680007762133,
						0.52720118893091983, 0.68845903945259213,
						0.28117234366042648, -0.24984642432648865,
						-0.19594627437659665, 0.12736934033574265,
						0.093057364603806592, -0.071394147165860775,
						-0.029457536821945671, 0.033212674058933238,
						0.0036065535669883944, -0.010733175482979604,
						0.0013953517469940798, 0.0019924052949908499,
						-0.00068585669500468248, -0.0001164668549943862,
						9.3588670001089845e-05, -1.3264203002354869e-05)
		{}
	};


	// factory for runtime object creations - throws if unable
	inline DWT* create_Daubechies(const size_t& _N)
	{
		switch (_N)
		{
			case 2: return new Daubechies<2>();
			case 3: return new Daubechies<3>();
			case 4: return new Daubechies<4>();
			case 5: return new Daubechies<5>();
			case 6: return new Daubechies<6>();
			case 7: return new Daubechies<7>();
			case 8: return new Daubechies<8>();
			case 9: return new Daubechies<9>();
			case 10: return new Daubechies<10>();

			default: break;
		}

		throw std::exception("Wavelet undefined, unable to instantiate object");
	}
	
}

namespace fwt = fast_wavelet_transform;

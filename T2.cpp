// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


// TEST #1 (HELLO WORLD)
// motivation: to test the fast wavelet transform implementation with theorem
// features: examples of parametric instantiation of wavelet filters
// test parameters in the main() code, choose wav filter, number of tests, source size...
// output type: console


#include "stdafx.h"
#include "DSPX_ann_def.h"
#include "DSPX_financial_convert.h"
#include "DSPX_financial_bar.h"
#include "DSPX_financial_data.h"
#include "DSPX_fast_wavelet_transform.h"


#define BARSFILE		"DATA\\H1_13_15.txt" // 2013.01.01 00:00 -> 2015.06.30 22:00


template <class FWT_type>
struct Q
{
	typedef ann::real_type						real_type;
	typedef ann::real_vector_type				real_vector_type;
	typedef ann::real_matrix_type				real_matrix_type;

	typedef FWT_type							transformer_type;
	typedef fwt::shift_variance_theorem			theorem_type;

	Q(const size_t& _DWTInputSz)
	: _InputSz(_DWTInputSz)
	, _DWT()
	, _Theorem(_DWTInputSz, _DWT.size()/2)
	{
		_retrieveVariantCoefficients();
		_retrieveSVTCoefficients();
		_retrieveSVTBacksteps();
	}

	~Q()
	{}


	auto source_size() const ->size_t {return _InputSz;} // horz (source) size

	auto history_size() const ->size_t {return _Transforms.size();}	// vertical Q size


	template <class _Ranit>
	void full_transform(const _Ranit& _Beg, const _Ranit& _End)
	{
		// fast wavelet tranform...
		// save a new crystal into matrix Q

		_Transforms.push_back(real_vector_type(source_size()));

		real_vector_type& _Out(*(_Transforms.rbegin()));
			
		_DWT.transform(_Beg._Ptr, &_Out[0], source_size());
	}

	template <class _Ranit>
	bool theorem_transform_test(const _Ranit& _Beg, const _Ranit& _End)
	{
		// Save a new DWT crystal into matrix Q, calculated using the shift variance theorem.
		// Test for algorithm correctness comparing the reduced transform crystal...
		// ...with the one the fast wavelet transfom

		_Transforms.push_back(real_vector_type(source_size()));

		real_vector_type& _Out(*(_Transforms.rbegin()));

		real_vector_type _Test(source_size());


		// perform reduced transform
		_DWT.transform(_VariantSizes, _TheoremBacksteps, _Transforms, _Beg._Ptr, &_Out[0], source_size());

		// perform fast wavelet transform
		_DWT.transform(_Beg._Ptr, &_Test[0], source_size());

		// test wavelet transform crystals are equal
		return (_Test == _Out);
	}

	template <class _Ranit>
	void theorem_transform(const _Ranit& _Beg, const _Ranit& _End)
	{// save a new DWT crystal into matrix Q, calculated using the shift variance theorem

		_Transforms.push_back(real_vector_type(source_size()));

		real_vector_type& _Out = *(_Transforms.rbegin());

		_DWT.transform(_VariantSizes, _TheoremBacksteps, 
			_Transforms, _Beg._Ptr, &_Out[0], source_size());
	}

private:

	void _retrieveVariantCoefficients()
	{
		_Theorem.variant_coefficients(_VariantSizes);
	}

	void _retrieveSVTCoefficients()
	{
		// retrieve all SVT coefficients ordinals

		for (size_t i=0; i<_InputSz; ++i) 
			if (_Theorem.is_SVT_coefficient(i)) 
				_TheoremCoefficients.insert(i);
	}

	void _retrieveSVTBacksteps()
	{
		// retrieve backsteps necessary to copy...
		// ...SVT coefficients from matrix Q

		for (size_t _N=_InputSz/2; 
				_N>=std::pow(2.0, std::ceil(std::log2(_DWT.size()))); 
					_N>>=1)

			_TheoremBacksteps.push_back(_Theorem.back_steps(_N));
	}


	size_t							_InputSz;					// eg. 128
	transformer_type				_DWT;						// wavelet transform object
	theorem_type					_Theorem;					// SVT theorem object
	std::unordered_set<size_t>		_TheoremCoefficients;		// Theorem coefficients ordinals
	std::vector<size_t>				_TheoremBacksteps;			// backsteps for theorem copy
	std::vector<size_t>				_VariantSizes;				// number of variant coefficients for each scale
	real_matrix_type				_Transforms;				// transforms history (matrix Q)
};

class stopwatch
{
	typedef std::chrono::steady_clock			clock_type;
	typedef std::chrono::time_point<clock_type>	time_point_type;
	typedef std::chrono::microseconds			duration_type;
	typedef typename duration_type::rep			rep_type;

public:

	stopwatch() 
		: _Start(time_point_type())
		, _Stop(time_point_type())
	{}

	~stopwatch() {}


	void start() {_now(_Start);}

	auto stop() ->stopwatch& {_now(_Stop); return *this;}

	auto elapsed() const ->rep_type {return std::chrono::duration_cast<duration_type>(_Stop - _Start).count();}

	auto reset() ->stopwatch&  {_Start=_Stop= time_point_type(); return *this;}

private:

	void _now(time_point_type& _Dest) {_Dest=std::chrono::steady_clock::now();}


	time_point_type		_Start, _Stop;
};


int main()
{
	// import typenames ...
	typedef ann::real_type					real_type;
	typedef ann::real_vector_type			vector_type;

	// parametric instantiation of wavelet filter type
	// pick one ...

	//typedef fwt::Daubechies<2>				FWT_type;
	//typedef fwt::Daubechies<3>				FWT_type;
	//typedef fwt::Daubechies<4>				FWT_type;
	//typedef fwt::Daubechies<5>				FWT_type;
	//typedef fwt::Daubechies<6>				FWT_type;
	//typedef fwt::Daubechies<7>				FWT_type;
	//typedef fwt::Daubechies<8>				FWT_type;
	//typedef fwt::Daubechies<9>				FWT_type;
	typedef fwt::Daubechies<10>				FWT_type;
	//typedef fwt::Daubechies<11>				FWT_type;	// !ops!


	// test parameters (choose)
	
	const size_t CORRECTNESS_TESTS(10);		// choose how many of these test you wanna do
	
	const size_t EFFICIENCY_TESTS(20000);	// choose how many of theese for speed test (max 20000 for source series size)

	const size_t PATSIZE(256);				// source series analyzing window size (e.g. 128, 256, 512 ...)



	// ----------->>>>>>>>>>

	const size_t QSIZE(PATSIZE);			// size of matrix Q

	// source file paths
	const path_type P(BARSFILE);

	// create data obj, load data
	financials::data DATA(P);

	// retrieve iterators, Bitcoin close price hourly (see DSPX_include.h)
	const auto _01_01_13 = DATA.close_it("2013.01.01 00:00"); // yy mm dd hh mm
	const auto _01_01_15 = DATA.close_it("2015.01.01 00:00"); // yy mm dd hh mm

	const auto CLOSEBEG = DATA.close_begin();
	const auto CLOSEEND = DATA.close_end();

	// check data loaded correctly or abort
	if (financials::_Failure(DATA)) return 0;

	// create Q object using helper class above
	Q<FWT_type> _Q(PATSIZE);

	// courtesy iterators
	vector_type::const_iterator BEG = CLOSEBEG;
	vector_type::const_iterator END = BEG + QSIZE;

	cout << "Theorem on the computational speed of the Reduced Wavelet Transform\n\n";

	// fill Q
	for (auto I = BEG; I < END; ++I) _Q.full_transform(I, I + PATSIZE);

	cout << "Q size: " << _Q.history_size() << "\n";


	BEG = END;
	END += CORRECTNESS_TESTS;

	// correctness of the reduced DWT algo test...

	for (auto I = BEG; I < END; ++I) 
		cout << ((_Q.theorem_transform_test(I, I + PATSIZE))? "Test correct":"Test failed") << "\n";


	BEG = END;
	END += EFFICIENCY_TESTS;

	 
	// speed tests...
	stopwatch SW;

	SW.start();
	for (auto I = BEG; I < END; ++I) _Q.full_transform(I, I + PATSIZE);	// FULL TRANSFORM
	cout << "Fast Wavelet Transform, Elapsed:\t" << SW.stop().elapsed() << "\n";


	SW.reset().start();
	for (auto I = BEG; I < END; ++I) _Q.theorem_transform(I, I + PATSIZE);	// REDUCED TRANSFORM
	cout << "Reduced Wavelet Transform, Elapsed:\t" << SW.stop().elapsed() << "\n";

	return 0;
}
// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


// TEST #1 (HELLO WORLD)
// motivation: to test the fast wavelet transform implementation
// features: code below shows how to create transformer objects creation at runtime
// output type: console

#include "stdafx.h"
#include "DSPX_ann_def.h"
#include "DSPX_ann_helper.h"
#include "DSPX_financial_convert.h"
#include "DSPX_financial_bar.h"
#include "DSPX_financial_data.h"
#include "DSPX_fast_wavelet_transform.h"

#define BARSFILE		"DATA\\H1_13_15.txt" // 2013.01.01 00:00 -> 2015.06.30 22:00


int main()
{
	typedef ann::real_type				real_type;
	typedef ann::real_vector_type		vector_type;
	typedef ann::real_matrix_type		matrix_type;

	// source file paths
	const path_type P(BARSFILE);

	// constants
	const size_t PATSIZE(128);	// source series analyzing window size
	const size_t MAXTEST(5);	// number of tests to perform for each analyzing wavelet

	// create data obj
	financials::data DATA(P);

	// retrieve iterators, Bitcoin close price hourly (see DSPX_include.h)
	const auto _01_01_13 = DATA.close_it("2013.01.01 00:00"); // yy mm dd hh mm
	const auto _01_01_14 = DATA.close_it("2014.01.01 00:00"); // yy mm dd hh mm
	const auto _01_01_15 = DATA.close_it("2015.01.01 00:00"); // yy mm dd hh mm

	const auto CLOSEBEG = DATA.close_begin();
	const auto CLOSEEND = DATA.close_end();

	const auto CLOSETRAINBEG = CLOSEBEG;
	const auto CLOSETRAINEND = _01_01_15;
	
	const auto CLOSETESTBEG = CLOSETRAINEND;
	const auto CLOSETESTEND = CLOSEEND;

	// map transformer objects with key= size of Daubechies wavelet support
	std::map<size_t, const fwt::DWT*> DMAP;

	// create transformer objects
	try 
	{
		for (size_t N = 2; N <= 10; ++N) DMAP[N] = fwt::create_Daubechies(N);
	}

	catch (std::exception xe) {cout << xe.what() << "\ntest failure\n"; return 0;}


	// test
	for (size_t N=2; N<=10; ++N) // for each Daubechies wavelet type
	{
		const fwt::DWT* D = DMAP[N];	// get ptr to transformer object
			
		vector_type::const_pointer _Data(CLOSETRAINBEG._Ptr);	// get ptr to source series

		for (size_t i=0; i<MAXTEST; ++i, ++_Data) 
		{
			vector_type _Forward(PATSIZE), _Reconstruction(PATSIZE), _AbsError(PATSIZE);

			D->transform(_Data, &_Forward[0], PATSIZE);	// discrete wavelet transform into _Forward

			D->invert(&_Forward[0], &_Reconstruction[0], PATSIZE); // inverse DWT into _Reconstruction


			// find reconstruction error

			for (size_t i=0; i<PATSIZE; ++i) _AbsError[i] = std::abs(*(_Data+i) - _Reconstruction[i]);

			const real_type _MeanAbsErr(ann::mean(_AbsError));
			

			// output reconstruction error

			cout << "Test#" << 1+i << ", " << D->wavelet_type() << ", Reconstruction MAE: " << _MeanAbsErr << "\n";
		}

		cout << "\n";
		
	}

	// cleanup dynamic mem
	for (size_t N = 2; N <= 10; ++N) {delete DMAP[N];}

	return 0;
}




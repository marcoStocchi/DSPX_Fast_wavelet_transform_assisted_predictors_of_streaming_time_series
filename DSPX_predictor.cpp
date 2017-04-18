// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// testing the case study

#include "stdafx.h"
#include "DSPX_ann_def.h"
#include "DSPX_ann_helper.h"
#include "DSPX_ann_neuron.h"
#include "DSPX_ann_layer.h"
#include "DSPX_ann_layer_input.h"
#include "DSPX_ann_neuron_perceptron.h"
#include "DSPX_ann_neuron_output.h"
#include "DSPX_ann_layer_perceptron.h"
#include "DSPX_ann_layer_output.h"
#include "DSPX_ann_network.h"
#include "DSPX_ann_network_perceptron.h"
#include "DSPX_ann_network_help.h"
#include "DSPX_financial_convert.h"
#include "DSPX_financial_bar.h"
#include "DSPX_financial_data.h"
#include "DSPX_fast_wavelet_transform.h"
#include "DSPX_help.h"
#include "DSPX_engine.h"

#define BARSFILE		"DATA\\H1_13_15.txt" // 2013.01.01 00:00 -> 2015.06.30 22:00



template <class engine_type, class _Init>
inline void _Create(engine_type& _Engine, const size_t& PATSIZE, const _Init& _Beg, const _Init& _End)
{
	cout << "creating Matrix Q\n";

	for (auto I = _Beg; I != _End; ++I)
	{
		// for each cycle, update engine
		_Engine.update(I, I+PATSIZE);
	}
}

template <class engine_type, class _Init>
inline void _Train(engine_type& _Engine, const size_t& PATSIZE, const _Init& _Beg, const _Init& _End)
{
	typedef predictor_system::real_type		real_type;

	cout << "training machines\n";

	real_type _LastKnown(0.0);

	// main testing loop
	for (auto I = _Beg; I != _End; ++I)
	{
		_Engine.predict(); // ignore prediction

		//---------UPDATE PHASE----------

		_LastKnown = *(_End-1); // last known series value

		_Engine.update(I, I+PATSIZE); // for each cycle, update engine

		cout << (I-_Beg) << "\r";
	}

	cout << "\n";
}

template <class engine_type, class _Init>
inline void _Test(engine_type& _Engine, const size_t& PATSIZE, 
	const _Init& _Beg, const _Init& _End,
		bool _BdumpQ, bool _BdumpFcast)
{
	typedef predictor_system::real_type	real_type;

	if (_Engine.trained()) cout << "Engine trained\n"; else { cout << "Failure to train engine"; return;}
	
	const size_t _PredictionAttempts(std::distance(_Beg, _End));

	cout << "test and retrain, " << _PredictionAttempts << " prediction attempts\n";

	// prep streams
	std::ofstream fout1, fout2;
	
	if (_BdumpQ) { fout1.open("crystals.txt"); predictor_system::set_stream(fout1); }

	if (_BdumpFcast) { fout2.open("results.txt"); predictor_system::set_stream(fout2); }


	real_type _MAE(0.0);
	
	real_type _LastKnown(0.0);

	// main testing loop
	for (auto I = _Beg; I != _End; ++I)
	{
		// get real value forecast
		real_type _Fcast=_Engine.predict(I, I+PATSIZE);// next pattern

		const real_type _Delta(std::abs(_Fcast - _LastKnown));


		//---------UPDATE PHASE-----------------

		// retrieve and store last series value
		_LastKnown = *(I+PATSIZE-1); 

		// find last absolute error
		const real_type _AbsErr(std::abs(_LastKnown - _Fcast));
		
		_MAE += _AbsErr;

		// update engine...
		_Engine.update(I, I+PATSIZE);


		// dump results...
		if (_BdumpQ) _Engine.dump_lastrow_nonSVT_diagnose(fout1);

		if (_BdumpFcast) fout2 << _LastKnown << "\t" << _Fcast << "\t" << _AbsErr << "\n";

		if ((I-_Beg)%10==0)cout << (I-_Beg) << "\r";
	}

	_MAE /= _PredictionAttempts;

	cout << "\nMAE:" << _MAE << "\n";
}


int main()
{
	// import typenames ...
	typedef predictor_system::real_type				real_type;
	typedef predictor_system::real_vector_type		vector_type;
	typedef predictor_system::real_matrix_type		matrix_type;
	typedef fwt::Daubechies<2>						FWT_type;
	typedef predictor_system::engine<FWT_type>		engine_type;

	// source file paths
	const path_type P(BARSFILE);

	// constants
	const size_t PATSIZE(128);			// source series analyzing window size
	const size_t QSIZE(PATSIZE);		// size of matrix Q

	// create data obj, load data
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

	// check data loaded correctly or abort
	if (financials::_Failure(DATA)) return 0;

	cout << "DATASET size:" << DATA.size() << "\n";


	// istantiate FWT object and predictor system...


	engine_type ENGINE(PATSIZE);

	ENGINE.dump_engine_diagnose(cout);

	const size_t _TrainingIterations(1*QSIZE);

	vector_type::const_iterator BEG = CLOSETRAINEND - _TrainingIterations - QSIZE;
	vector_type::const_iterator END = BEG + QSIZE;

	_Create(ENGINE, PATSIZE, BEG, END); // creates Q matrix

	BEG = END;
	END = BEG + _TrainingIterations;

	_Train(ENGINE, PATSIZE, BEG, END); // trains networks

	BEG = END;
	//END = BEG + 20*QSIZE;
	END = CLOSETESTEND - PATSIZE;

	// test and retrain predictor
	// ... dump forecast results
	_Test(ENGINE, PATSIZE, BEG, END, true/*Q output*/, true /*results output*/); 
	

	return 0;
}



// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// testing the case study - benchmark SVM 

#include "stdafx.h"
#include "DSPX_ann_def.h"
#include "DSPX_ann_helper.h"
#include "DSPX_financial_convert.h"
#include "DSPX_financial_bar.h"
#include "DSPX_financial_data.h"
#include "DSPX_help.h"


#include <dlib-18.18\dlib\svm.h>

#define BARSFILE		"DATA\\H1_13_15.txt" // 2013.01.01 00:00 -> 2015.06.30 22:00


template <class engine_type, class _Init>
inline void _Train(engine_type& _Engine, const size_t& PATSIZE, const _Init& _Beg, const _Init& _End)
{
	typedef ann::real_type							real_type;
	typedef ann::real_vector_type					real_vector_type;
    typedef dlib::matrix<real_type,1,1>				sample_type;
	typedef dlib::radial_basis_kernel<sample_type>	kernel_type;

	sample_type m;

	cout << "training machines\n";
 

	// main loop
	for (auto I = _Beg; I != _End; ++I)
	{
		std::vector<sample_type> samples;
		real_vector_type targets;

		for (size_t i=0; i<PATSIZE; ++i)
		{
			m(0) = i;

			samples.push_back(m);

			targets.push_back(*(I+i));
		}

		// ignore results
		_Engine.train(samples, targets);

		cout << (I-_Beg) << "\r";
	}

	cout << "\nNetwork trained\n"; 
}

template <class engine_type, class _Init>
inline void _Test(engine_type& _Engine, const size_t& PATSIZE, 
	const _Init& _Beg, const _Init& _End,
		bool _BdumpFcast)
{
	typedef ann::real_type							real_type;
	typedef ann::real_vector_type					real_vector_type;
    typedef dlib::matrix<real_type,1,1>				sample_type;
	typedef dlib::radial_basis_kernel<sample_type>	kernel_type;

		
	const size_t _PredictionAttempts(std::distance(_Beg, _End));

	cout << "test and retrain, " << _PredictionAttempts << " prediction attempts\n";

	// prep streams
	std::ofstream fout;
	
	if (_BdumpFcast) { fout.open("bench_results_SVM.txt"); predictor_system::set_stream(fout); }

	sample_type m;

	real_type _MAE(0.0);
	
	real_type _LastKnown(0.0);

	// main testing loop
	for (auto I = _Beg; I != _End; ++I)
	{

		std::vector<sample_type> samples;

		real_vector_type targets;

		for (size_t i=0; i<PATSIZE; ++i)
		{
			m(0) = i;

			samples.push_back(m);

			targets.push_back(*(I+i-1));
		}

		// training 
		dlib::decision_function<kernel_type> df = _Engine.train(samples, targets);

		// get real value forecast
		m(0)=PATSIZE;

		real_type _Fcast = df(m);
		

		// retrieve and store last series value
		_LastKnown = *(I+PATSIZE-1); 

		// find last absolute error
		const real_type _AbsErr(std::abs(_LastKnown - _Fcast));
		
		_MAE += _AbsErr;


		// dump results...

		if (_BdumpFcast) fout << _LastKnown << "\t" << _Fcast << "\t" << _AbsErr << "\n";

		if ((I-_Beg)%10==0)cout << (I-_Beg) << "\r";
	}

	_MAE /= _PredictionAttempts;

	cout << "\nMAE:" << _MAE << "\n";
}

int main()
{
	// import typenames ...
	typedef ann::real_type							real_type;
	typedef ann::real_vector_type					vector_type;
    typedef dlib::matrix<double,1,1>				sample_type;
    typedef dlib::radial_basis_kernel<sample_type>	kernel_type;



	// source file paths
	const path_type P(BARSFILE);

	// constants
	const size_t MARGIN(128);			// constant used to benchmark purposes
	const size_t INPUTSIZE(32);			// input size
	const real_type _SVMK(0.1);			// SVM kernel 
	const real_type	_SVMC(10);			// SVM regularization
	const real_type	_SVME(0.001);		// SVM epsilon insensitivity

	const size_t _TrainingIterations(MARGIN);


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
	

 
    dlib::svr_trainer<kernel_type> _Engine;

    _Engine.set_kernel(kernel_type(_SVMK));

    // This parameter is the usual regularization parameter.  It determines the trade-off 
    // between trying to reduce the training error or allowing more errors but hopefully 
    // improving the generalization of the resulting function.  Larger values encourage exact 
    // fitting while smaller values of C may encourage better generalization.
    _Engine.set_c(_SVMC);
    // Epsilon-insensitive regression means we do regression but stop trying to fit a data 
    // point once it is "close enough" to its target value.  This parameter is the value that 
    // controls what we mean by "close enough". 
    _Engine.set_epsilon_insensitivity(_SVME);



	vector_type::const_iterator BEG = CLOSETRAINEND - _TrainingIterations;
	vector_type::const_iterator END = CLOSETRAINEND;


	_Train(_Engine, INPUTSIZE, BEG, END); // trains networks

	BEG = END;
	//END = BEG + 20*QSIZE;
	END = CLOSETESTEND - INPUTSIZE;

	// test and retrain predictor
	// ... dump forecast results
	_Test(_Engine, INPUTSIZE, BEG, END, true /*results output*/); 
	

	return 0;
}



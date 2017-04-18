// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// testing the case study - benchmark NNetwork

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

template <class mlp_type>
class neural_predictor_t
{
	typedef predictor_system::real_type		value_type;

public:

	template <class... _Sizes>
	neural_predictor_t(_Sizes... _Args)
		: _mlp(_Args...)
	{}


	void set_mlp_learningrate(const value_type& _Lrate) 
	{// set mlp learning rate param
		_mlp.set_learning_rate(_Lrate);
	}

	void set_mlp_mM_errors(const value_type& M, const value_type& m) 
	{// set min max errors to mlp retraining purposes
		_MaxErr=M; _MinErr =m;
	}

	template <class _Init>
	auto predict (const _Init& _Beg, const _Init& _End) ->value_type
	{// use first differences			


		// input vector for first differences of the source series
		std::vector<value_type> _dinput;

		// extract first differences
		for (auto I=_Beg; I!=_End;++I) _dinput.push_back(*I-*(I-1));
				
		//cout << _dinput.size() << "\n";

		// test mlp with input
		const value_type _sdFcst = network_test_single(_mlp, _dinput);

		// invert sigmoid
		const value_type _dFcst = ann::hyperbolic_tangent::invert(_sdFcst);

		// store last 1stdiff sigmoided result
		_Last_sdFcst = _sdFcst; // value used in the next update()

		// revert from 1st differences to real value
		const value_type _Fcst = _dFcst + *(_End-1);
				
		return _Fcst;
	}

	template <class _Init>
	void update(const _Init& _Beg, const _Init& _End)
	{// last coefficient already updated


		// input vec for first differences of the source series
		std::vector<value_type> _dinput;

		// extract first differences
		for (auto I=_Beg; I!=_End;++I) _dinput.push_back(*I-*(I-1));

		// cache actual last 1st diff value
		const value_type _dActual(*_dinput.crbegin());

		// get sigmoided actual value
		const value_type _sdActual = ann::hyperbolic_tangent::execute(_dActual);

		// get sigmoided last prediction error 
		const value_type _Err = _Last_sdFcst - _sdActual;
				
		// train network if minErr has been violated
		network_train_single(_mlp, _dinput, _Err, _MaxErr, _MinErr, _sdActual);
	}

private:

	mlp_type			_mlp;
	value_type			_Last_sdFcst;	// depot 
	value_type			_MaxErr;
	value_type			_MinErr;
};

template <class engine_type, class _Init>
inline void _Train(engine_type& _Engine, const size_t& PATSIZE, const _Init& _Beg, const _Init& _End)
{
	typedef predictor_system::real_type		real_type;

	cout << "training machines\n";

	real_type _LastKnown(0.0);

	// main testing loop
	for (auto I = _Beg; I != _End; ++I)
	{
		_Engine.predict(I-1, I+PATSIZE-1); // ignore prediction

		//---------UPDATE PHASE----------

		_LastKnown = *(_End-1); // last known series value

		_Engine.update(I, I+PATSIZE); // for each cycle, update engine

		cout << (I-_Beg) << "\r";
	}

	cout << "\nNetwork trained\n"; 
}

template <class engine_type, class _Init>
inline void _Test(engine_type& _Engine, const size_t& PATSIZE, 
	const _Init& _Beg, const _Init& _End,
		bool _BdumpFcast)
{
	typedef predictor_system::real_type	real_type;

		
	const size_t _PredictionAttempts(std::distance(_Beg, _End));

	cout << "test and retrain, " << _PredictionAttempts << " prediction attempts\n";

	// prep streams
	std::ofstream fout;
	
	if (_BdumpFcast) { fout.open("bench_results_ANN.txt"); predictor_system::set_stream(fout); }


	real_type _MAE(0.0);
	
	real_type _LastKnown(0.0);

	// main testing loop
	for (auto I = _Beg; I != _End; ++I)
	{
		// get real value forecast
		real_type _Fcast=_Engine.predict(I-1, I+PATSIZE-1);// next pattern

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

		if (_BdumpFcast) fout << _LastKnown << "\t" << _Fcast << "\t" << _AbsErr << "\n";

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
	typedef predictor_system::m1lp_type				neural_network_type;
	typedef neural_predictor_t<neural_network_type>	neural_predictor_type;


	// source file paths
	const path_type P(BARSFILE);

	// constants
	const size_t MARGIN(128);			// constant used to benchmark purposes
	const size_t NEURALINPUTSIZE(32);	// neural network input size
	const real_type _MaxErr(.001);		// neural network testing maximum error
	const real_type _MinErr(.000001);	// neural network retrainin minimum error
	const real_type _LearningRate(0.1);	// ...
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
	

	//neural_predictor_type _Engine(NEURALINPUTSIZE , 2*NEURALINPUTSIZE , 1);
	neural_predictor_type _Engine(
		NEURALINPUTSIZE, 
			2*NEURALINPUTSIZE, 
				1);

	_Engine.set_mlp_learningrate(_LearningRate);

	_Engine.set_mlp_mM_errors(_MaxErr, _MinErr);


	vector_type::const_iterator BEG = CLOSETRAINEND - _TrainingIterations;
	vector_type::const_iterator END = CLOSETRAINEND;


	_Train(_Engine, NEURALINPUTSIZE, BEG, END); // trains networks

	BEG = END;
	//END = BEG + 20*QSIZE;
	END = CLOSETESTEND - NEURALINPUTSIZE;//MARGIN;

	// test and retrain predictor
	// ... dump forecast results
	_Test(_Engine, NEURALINPUTSIZE, BEG, END, true /*results output*/); 
	

	return 0;
}



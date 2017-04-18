// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#pragma once
#define M01 "theorem failed, not enough history dwts"
#define M02 "neuralnet failed, not enough history dwts"

// 
// FWT assisted inference engine for streaming datasets
// tobedone:
// 

namespace predictor_system
{
	typedef ann::real_type											real_type;
	typedef ann::real_vector_type									real_vector_type;
	typedef ann::real_matrix_type									real_matrix_type;

	typedef fwt::shift_variance_theorem								shift_variance_theorem;

	typedef ann::general_multi_layer_perceptron<
			ann::hyperbolic_tangent, ann::input_layer, 
				ann::perceptron_layer, 
					ann::output_layer>								m1lp_type; // single hidden layer MLP

	typedef ann::general_multi_layer_perceptron<
			ann::hyperbolic_tangent, ann::input_layer, 
				ann::perceptron_layer, 
				ann::perceptron_layer, 
				ann::perceptron_layer, 
					ann::output_layer>								m3lp_type; // triple hidden layer MLP

	template <class matrix_type>
	class predictor
	{// abstract class for virtual predictors

	protected:

		typedef typename matrix_type::value_type::value_type		value_type;

	public:

		virtual ~predictor() {}

		virtual auto predict(const matrix_type& _M, size_t i) ->value_type = 0;

		virtual void update(const matrix_type& _M, size_t i) = 0;
	};

	namespace /*...predictors*/
	{
		template <class matrix_type,  class _PredictorT>
		class predictor_spec /*undef*/;

		template <class matrix_type>
		class predictor_spec <matrix_type, m1lp_type> : public predictor <matrix_type>
		{// specialization for single hidden layer perceptron networks

		public:

			template <class... sizes>
			predictor_spec(sizes... i)
				: _mlp(i...)
				, _Last_sdFcst(0.0)
				, _MaxErr(0)	// lazy set
				, _MinErr(0)	// ...
			{
			}

			~predictor_spec() {}


			void set_mlp_learningrate(const value_type& _Lrate) 
			{// set mlp learning rate param
				_mlp.set_learning_rate(_Lrate);
			}

			void set_mlp_mM_errors(const value_type& M, const value_type& m) 
			{// set min max errors to mlp retraining purposes
				_MaxErr=M; _MinErr =m;
			}

			virtual auto predict /*throws*/(const matrix_type& _M, size_t i) ->value_type
			{// use first differences			

				// cache hist and input sizes
				const size_t _history_size(_M.size()), _input_size(_mlp.input_size());

				// precheck this...
				if (_input_size + 1 > _history_size) throw std::exception(M02);

				// find indeces...
				const size_t _vecbeg(_history_size - _input_size), _vecend(_history_size);

				// allocate input for first differences of the source series
				std::vector<value_type> _dinput(_input_size);

				// extract first differences
				for (size_t z=0, vi=_vecbeg, ve=_vecend; vi<ve; ++vi, ++z) _dinput[z] = _M[vi][i] - _M[vi - 1][i];
				
				// test mlp with input
				const value_type _sdFcst = network_test_single(_mlp, _dinput);

				// invert sigmoid
				const value_type _dFcst = ann::hyperbolic_tangent::invert(_sdFcst);

				// store last 1stdiff sigmoided result
				_Last_sdFcst = _sdFcst; // value used in the next update()

				// revert from 1st differences to real value
				const value_type _Fcst = _dFcst + _M[_vecend - 1][i]; 
				
				return _Fcst;
			}

			virtual void update(const matrix_type& _M, size_t i)
			{// last coefficient already updated

				// cache hist and input sizes
				const size_t _history_size(_M.size()), _input_size(_mlp.input_size());
				
				// precheck this...
				if (_input_size + 1 > _history_size) throw std::exception(M02);

				// find indeces...
				const size_t _vecbeg(_history_size - _input_size), _vecend(_history_size);
				
				// allocate input for first differences of the source series
				std::vector<value_type> _dinput(_input_size);

				// extract first differences
				for (size_t z = 0, vi = _vecbeg, ve = _vecend; vi < ve; ++vi, ++z) _dinput[z] = _M[vi][i] - _M[vi - 1][i];
				
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

			m1lp_type			_mlp;			// neural network

			value_type			_Last_sdFcst;	// depot 
			value_type			_MaxErr;
			value_type			_MinErr;
		};

		template <class matrix_type>
		class predictor_spec <matrix_type, shift_variance_theorem> : public predictor <matrix_type>
		{// specialization for theorem coefficients transposition

		public:

			predictor_spec(const shift_variance_theorem& _Th)
				: _Theorem(_Th)
			{}

			~predictor_spec() {}


			virtual auto predict /*throws*/(const matrix_type& _M, size_t i) ->value_type
			{
				const size_t _backsteps(_Theorem.back_steps(i)), _history_size(_M.size());

				// precheck this...
				if (_history_size < _backsteps + 1) throw std::exception(M01);

				// return ritish column transposed coefficient...
				return _M[_history_size - _backsteps][i + 1];
			}

			virtual void update(const matrix_type& _M, size_t i) {/*donothing*/ }

		private:

			const shift_variance_theorem&			_Theorem;
		};	

		// develop other predictor_spec here...
	}
	
	template <class matrix_type>
	class predictor_container
	{// wrapper object to hold virtual predictor objects

		typedef typename matrix_type::value_type::value_type			value_type;
		typedef predictor <matrix_type>									predictor;
		typedef predictor_spec<matrix_type, m1lp_type>					neural_predictor_type;
		typedef predictor_spec<matrix_type, shift_variance_theorem>		theorem_predictor_type;
		
		// ... import other predictor_spec specialization types here

	public:

		typedef predictor												predictor_type;


		predictor_container(const shift_variance_theorem& _Th)
		{
			_default_create_predictors(_Th);
		}

		~predictor_container()
		{
			_destroy_predictors();
		}


		predictor* operator[] (const size_t& i) const { return _Prd.at(i); }

	private:

		void _default_create_predictors(const shift_variance_theorem& _Th)
		{

			for (size_t i = 0; i < _Th.source_size(); ++i)
			{
				if (_Th.is_SVT_coefficient(i)) _Prd[i] = new theorem_predictor_type(_Th);

				else // MLP, SOM/SOL, SVM, compound, etc. 
				{// e.g. Daub4 -> 5 6 7 - 13 14 15 - 29 30 31 - 61 62 63 - 126 127
					
					const size_t __NEURALINPUTSIZE = 8;

					neural_predictor_type* ptr = 
						new neural_predictor_type(__NEURALINPUTSIZE , 2*__NEURALINPUTSIZE , 1);
						
					//const value_type _MaxErr(.0001), _MinErr(.00001);
					const value_type _MaxErr(.01), _MinErr(.000001);

					ptr->set_mlp_learningrate(0.1);

					ptr->set_mlp_mM_errors(_MaxErr, _MinErr);

					_Prd[i] = ptr;
				}
			}
		}

		void _destroy_predictors()
		{
			for (auto I = _Prd.begin(), E = _Prd.end(); I != E; ++I)
			{
				safe_delete(I->second);
			}
		}


		std::map<size_t, predictor*>		_Prd;		// mapped predictors
	};

	template <class FWT_type>
	class engine
	{
		typedef real_type									value_type;
		typedef real_vector_type							vector_type;
		typedef real_matrix_type							matrix_type;

		typedef FWT_type									transformer_type;
		typedef fwt::shift_variance_theorem					theorem_type;
		typedef predictor_container<matrix_type>			predictor_container_type;
		typedef predictor_container_type::predictor_type	predictor_type;

	public:

		engine(const size_t& _DWTInputSz)
			: _InputSz(_DWTInputSz)
			, _DWT()
			, _Theorem(_DWTInputSz, _DWT.size()/2)
			, _Transforms()
			, _Forecasts()
			, _Sources()
			, _Inverted()
			, _Predictors(_Theorem) // creates predictors
			, _Fcst(source_size()) // allocate
			, _Inv(source_size()) // ...
		{

		}

		~engine()
		{}


		bool trained() const {return _Forecasts.size()==_Transforms.size();} // predictors trained

		auto source_size() const ->size_t {return _InputSz;}

		auto history_size() const ->size_t {return _Transforms.size();}

		auto minQ_size() const ->size_t {return _InputSz;}

		auto predict()->value_type
		{
			// perform reduced FWT using the SVT theorem
			_reduce_predict();

			// get a reference to the crystal
			vector_type& _Out = *(--_Forecasts.end());

			// perform inverse DWT on it
			_DWT.invert(&_Out[0], &_Inv[0], source_size());
			
			// trim excess forecast vector from the storage
			if (_Forecasts.size() > minQ_size()) _Forecasts.erase(_Forecasts.begin());
			
			// store inverse DWT (forecasted series)
			_Inverted.push_back(_Inv);

			// trim...
			if (_Inverted.size() > minQ_size()) _Inverted.erase(_Inverted.begin());
			
			// return last element of the inverted DWT
			return *_Inv.crbegin();
		}

		template <class _Init>
		auto predict(const _Init& _Beg, const _Init& _End)->value_type
		{
			// perform reduced FWT using the SVT theorem
			_reduce_predict();

			// get a reference to the crystal
			vector_type& _Out = *(--_Forecasts.end());

			// optimize crystal 
			_optimize(_Beg, _End, _Out);

			// inverse DWT
			_DWT.invert(&_Out[0], &_Inv[0], source_size());

			// trim excess forecast vector from the storage
			if (_Forecasts.size() > minQ_size()) _Forecasts.erase(_Forecasts.begin());
			
			// store inverse DWT (forecasted series)
			_Inverted.push_back(_Inv);

			// trim...
			if (_Inverted.size() > minQ_size()) _Inverted.erase(_Inverted.begin());
			
			// return last element of the inverted DWT
			return *_Inv.crbegin();
		}

		template <class _Init>
		auto update(const _Init& _Beg, const _Init& _End)
		{// push-pop a new value in the source queue, transform and store the new DWT

			// pattern discrete wavelet transform 
			_full_transform(_Beg, _End);
			
			// not enough history in Q...
			if (history_size() <= minQ_size()) return;

			// trim excess vector in Q front...
			_Transforms.erase(_Transforms.begin());


			// for each ordinal
			for (size_t i = 0; i < source_size(); ++i)
			{
				// retrieve pointer to predictor
				predictor_type* ptr = _Predictors[i];
				
				// retrain predictor
				ptr->update(_Transforms, i);
			}
		}


		// diagnostic outputs

		void dump_engine_diagnose(std::ostream& s) const
		{
			_dump_nonSVT_coefficients(s, _InputSz);
		}

		void dump_lastrow_diagnose(std::ostream& s) const
		{
			const size_t _Ordinal(_Forecasts.size()-1);

			for (size_t i=0; i<_Transforms[_Ordinal].size(); ++i) 
				s << _Transforms[_Ordinal][i] << "\t" << _Forecasts[_Ordinal][i] << "\t\t";

			s << "\n";
		}

		void dump_lastrow_nonSVT_diagnose(std::ostream& s) const
		{
			const size_t _LastRow(_Forecasts.size()-1);

			for (size_t i=0; i<_Transforms[_LastRow].size(); ++i) 
				if (!_Theorem.is_SVT_coefficient(i))
					s << _Transforms[_LastRow][i] << "\t" << _Forecasts[_LastRow][i] << "\t\t";

			s << "\n";
		}

	private:

		template <class _Init>
		void _optimize(const _Init& _Beg, const _Init& _End, vector_type& _Out)
		{
			const size_t SZ = source_size();

			vector_type _CSrc(SZ), _CFcst1(SZ), _CFcst2(SZ);

			std::copy(_Beg, _End, _CSrc.begin());

			const value_type _Delta = 2.0;

			_CSrc[_CSrc.size() - 1] = _CSrc[_CSrc.size() - 2] - _Delta;
			
			_DWT.transform(&_CSrc[0], &_CFcst1[0], SZ); // fwd transform 1

			_CSrc[_CSrc.size() - 1] = _CSrc[_CSrc.size() - 2] ; // repeat last known series value

			_DWT.transform(&_CSrc[0], &_CFcst2[0], SZ); // fwd transform 2



			std::map<size_t, value_type> alphas, betas, xs; // linear equation, Y=alpha*X +beta


			for (size_t i = 0; i < SZ; ++i)
			{// find slope and intersection
				if (!_Theorem.is_SVT_coefficient(i))
				{
					alphas[i] = (_CFcst2[i] - _CFcst1[i]) / _Delta;
					betas[i] = _CFcst2[i];
				}
			}


			for (size_t i = 0; i < SZ; ++i)
			{// find Xs for each coefficient
				if (!_Theorem.is_SVT_coefficient(i))
				{ 
					xs[i] /*create*/ = (_Out[i] - betas.at(i))/ alphas.at(i);
				}
			}

			// filter outlier xs values
			real_vector_type VX;

			for (auto XI=xs.cbegin(), XE=xs.cend(); XI!=XE;++XI)
				if (std::abs(XI->second)<2) VX.push_back(XI->second);

			// find aritmetic mean of filtered Xs
			real_type X(ann::mean(VX));
			
			// optimize non-SVT coefficients...
			for (size_t i = 0; i < SZ; ++i)
			{
				if (!_Theorem.is_SVT_coefficient(i))
				{
					_Out[i]= alphas[i]* X + betas[i];
				}
			}
		}


		void _reduce_predict()
		{// forecast a new DWT crystal 

			_Forecasts.push_back(vector_type(source_size()));

			vector_type& _Out = *(--_Forecasts.end());
			
			// for each ordinal
			for (size_t i = 0; i < source_size(); ++i)
			{
				// test predictor and store forecasted DWT coefficient
				_Out[i] = _Predictors[i]->predict(_Transforms, i);
			}
		}

		template <class _Init>
		void _full_transform(const _Init& _Beg, const _Init& _End)
		{// save a new DWT crystal into matrix Q

			_Transforms.push_back(vector_type(source_size()));

			vector_type& _Out = *(--_Transforms.end());
			
			_DWT.transform(_Beg._Ptr, &_Out[0], source_size());
		}

		void _dump_nonSVT_coefficients(std::ostream& s, const size_t& _Srcsize) const
		{
			s << "variant coeff. ordinals: ";

			for (size_t i = 0; i < _Srcsize; ++i)
			{
				if (!_Theorem.is_SVT_coefficient(i))
					s << i << " ";
			}

			s << "\n";
		}


		size_t								_InputSz;			// e.g. 128
		transformer_type					_DWT;				// wavelet transform object
		theorem_type						_Theorem;			// Theorem object
		
		matrix_type							_Transforms;		// transforms history (matrix Q)
		matrix_type							_Forecasts;			// forecasted transforms history

		matrix_type							_Sources;			// actual pattern history 
		matrix_type							_Inverted;			// inverted transforms of forecasts, history 

		predictor_container_type			_Predictors;		// predictor container wrapper and factory
		vector_type							_Fcst;				// depot vector
		vector_type							_Inv;				// depot vector

	};
}
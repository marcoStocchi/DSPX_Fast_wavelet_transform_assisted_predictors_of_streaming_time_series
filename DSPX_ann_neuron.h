// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once


namespace artificial_neural_networks
{
	struct random_initializer
	{
		template <class C>
		static void initialize(C& _Cont, const size_t& _Sz, const real_type& _m, const real_type& _M)
		{
			_UninitializedRand(_Cont, _Sz, _m, _M);
		}

		static void initialize(real_type& _Val, const real_type& _m, const real_type& _M)
		{
			_UninitializedRand(_Val, _m, _M);
		}
	};

	template <class T/*value type*/, class _InitFunc /*activation functor*/>
	class neuron_t /*undefined*/; // base template for artificial neurons

	template <class _InitFunc>
	class neuron_t <real_type, _InitFunc>
	{
		// template specialization for real_type

	public:

		typedef unsigned char									byte;
		typedef real_vector_type								weights_container;		// store weights
		typedef typename weights_container::const_iterator		weights_const_iterator;
		typedef typename weights_container::iterator			weights_iterator;
		typedef typename weights_container::reverse_iterator	weights_reverse_iterator;
		

		neuron_t() 
		: _biasv(0)
		{}

		~ neuron_t() {}



		virtual auto output() const->real_type =0;


		// initialize weights

		void bias_initialize(const real_type& _m, const real_type& _M) { _InitFunc::initialize(_biasv, _m, _M); }

		void weights_initialize(size_t sz, const real_type& _m, const real_type& _M) { _InitFunc::initialize(_weights, sz, _m, _M); }


		// dump

		void dump_weights(std::ostream& s, char _Endl='\n') const
		{
			//s << std::setprecision(2);

			for (auto I= weights_begin(), E= weights_end(); I!=E;)
			{
				s << *I; if (++I!=weights_end()) s << " ";
			}
			s << _Endl;
		}

		void dump_bias(std::ostream& s, char _Endl='\n') const { s << _biasv << _Endl; }
		
		void binary_write(std::ofstream& fout) const
		{
			static const size_t _Frsz(sizeof(real_type));

			for (size_t i=0, e=_weights_size(); i<e; ++i)
			{
				const char* ptr = reinterpret_cast<const char*> (&_weights[i]);

				fout.write(ptr, _Frsz);
			}

			const char* ptr = reinterpret_cast<const char*> (&_biasv);

			fout.write(ptr, _Frsz);
		}

		bool binary_read(std::ifstream& fin, const size_t& n)
		{
			_weights.clear(); _weights.resize(n);

			static const size_t _Frsz(sizeof(real_type));

			for (size_t i=0; i<n; ++i)
			{
				char* ptr = reinterpret_cast<char*>(&_weights[i]);

				fin.read(ptr, _Frsz);
			}

			char* ptr = reinterpret_cast<char*>(&_biasv);

			fin.read(ptr, _Frsz);

			return fin.good();
		}

	protected:


		auto _weights_size() const ->size_t { return _weights.size(); }

		auto _weights_at(size_t i) const ->const real_type&{ return _weights[i]; }

		auto _weights_at(size_t i) ->real_type& { return _weights[i]; }

		auto _weights_cbegin() const ->weights_const_iterator { return _weights.cbegin(); }

		auto _weights_begin() ->weights_iterator { return _weights.begin(); }

		auto _weights_cend() const ->weights_const_iterator { return _weights.cend(); }

		auto _weights_end() ->weights_iterator { return _weights.end(); }

		auto _bias() const ->const real_type&{ return _biasv; }

		auto _bias() ->real_type& { return _biasv; }


	private:

		real_type				_biasv;
		weights_container		_weights;
	};


	typedef neuron_t<real_type, random_initializer>		real_random_initializer_neuron;
}
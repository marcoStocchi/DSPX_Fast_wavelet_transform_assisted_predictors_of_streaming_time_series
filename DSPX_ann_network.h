// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

namespace artificial_neural_networks
{

	template <class... args>
	class network
	{
		// base class for neural networks
		// provides protected functions to retrieval of layers of different typenames
		// provides protected functions for network state's storage

	public:
		// export typenames...
		typedef std::tuple<args...>										tuple_type;
		typedef typename std::tuple_size<tuple_type>					tuple_size_type;
		typedef network<args...>										network_type;
		typedef typename std::tuple_element<0, tuple_type>::type		input_layer_type;
		typedef typename std::tuple_element<tuple_size_type::value-1, tuple_type>::type		
																		output_layer_type;

		template <class... _Sizes>
		network(_Sizes... _Sz)
			: _layers(_Sz...)
		{
			//cout << tuple_size_type << "\n";
		}

		~network() {}

	protected:

		template <size_t i>
		auto _layer() -> typename std::tuple_element<i, tuple_type>::type& { return std::get<i>(_layers); }

		template <size_t i>
		auto _layer() const -> typename const std::tuple_element<i, tuple_type>::type&{ return std::get<i>(_layers); }

		auto _input_layer() ->input_layer_type& { return _layer<0>(); }

		auto _input_layer() const -> const input_layer_type&{ return _layer<0>(); }

		auto _output_layer() ->output_layer_type& { return _layer<tuple_size_type::value-1>(); }

		auto _output_layer() const -> const output_layer_type&{ return _layer<tuple_size_type::value-1>(); }


		bool _load(const path_type& P)
		{
			std::ifstream fin(P.string(), std::ios::binary);

			if (!fin.good()) return false;

			std::vector<size_t> layersizes(tuple_size_type::value); // contains input layer size
			
			for (size_t i=0; i<layersizes.size(); ++i)
			{
				char* ptr = reinterpret_cast<char*>(&layersizes[i]);

				fin.read(ptr, sizeof(size_t));
			}


			if (!fin.good()) return false;

			_load<1>(fin, layersizes);

			return fin.good();
		}

		void _save(const path_type& P) const
		{
			std::ofstream fout(P.string(), std::ios::binary);

			_savesize<0>(fout); // save input size too

			_save<1>(fout);
		}

		template <size_t I>
		void _save(std::ofstream& fout) const { _layer<I>().save(fout); return _save<I+1>(fout); }

		template <>
		void _save <tuple_size_type::value> (std::ofstream& fout) const { }

		template <size_t I>
		void _savesize(std::ofstream& fout) const 
		{ 
			size_t sz = _layer<I>().size(); 

			const char* ptr = reinterpret_cast<const char*> (&sz);

			fout.write(ptr, sizeof(size_t));

			return _savesize<I+1>(fout);
		}

		template <>
		void _savesize <tuple_size_type::value>(std::ofstream& fout) const {}

		void _dump(const path_type& P) const
		{
			std::ofstream fout(P.string()); fout << std::fixed;
			
			fout << "NETWORK INNER STATE DUMP\n";

			_dump<1>(fout);
		}

		template <size_t I>
		void _dump(std::ofstream& fout) const 
		{
			_layer<I>().dump(I, fout); return _dump<I+1>(fout); 
		}

		template <>
		void _dump <tuple_size_type::value> (std::ofstream& fout) const 
		{}


		template <size_t I>
		void _load(std::ifstream& fin, const std::vector<size_t>& _layersizes)
		{
			size_t _Prevsz = _layersizes[I-1];

			size_t _Thissz = _layersizes[I];

			_layer<I>().load(fin, _Thissz, _Prevsz);

			_load<I+1>(fin, _layersizes);
		}

		template <>
		void _load<tuple_size_type::value>(std::ifstream& fin, const std::vector<size_t>& _layersizes) 
		{}

	private:

		tuple_type		_layers;				// tuple of layers
	};

}
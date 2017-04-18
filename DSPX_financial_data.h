// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once
#define MSGERRORDATA	"Unable to load data"
#define MSGERROREXIT	"Aborting..."

namespace financials
{

	class data
	{
		// load bars archive, extract close prices, provides iterators...

		typedef double						value_type;
		typedef value_type*					pointer;
		typedef const value_type*			const_pointer;
		typedef std::vector<value_type>		vector_type;
		typedef std::vector<vector_type>	matrix_type;

	public:


		data(const path_type& _P)
			: _Good(financials::load_bars(_P, _Bars))
		{
	
			if (good())
			{
				_Closes.resize(_Bars.size());

				_extract_close_price(_Closes);
			}
		}

		~data() {}


		// test...

		bool good() const {return _Good;}


		// queries...

		auto size() const ->size_t { return _Closes.size(); } // size of bars

		auto time_to_index(const time_t& _Unixt) const ->size_t
		{// bar index 

			size_t i(0);

			for (const size_t e=size(); i<e; i+=100) // gross search
				if (_index_to_time(i) > _Unixt) break;

			if (!i) return 0;

			while (_index_to_time(--i) > _Unixt);

			return i+1;
		}

		auto time_to_index(const std::string& _DaytimeGroup) const ->size_t
		{// bar index 

			convert::time_string TS(_DaytimeGroup);

			const time_t UNIXTS = TS.to_unix_time();

			return this->time_to_index(UNIXTS);
		}

		auto index_to_unixtime(const size_t& _Idx) const ->size_t
		{
			return _index_to_time(_Idx);
		}

		auto index_to_strtime(size_t _Idx) const ->std::string
		{
			convert::time_string TS(_index_to_time(_Idx));

			return TS.to_datetime();
		}


		auto close_begin() const ->vector_type::const_iterator { return _Closes.cbegin(); }

		auto close_end() const ->vector_type::const_iterator { return _Closes.cend(); }

		auto close_it(const std::string& _DaytimeGroup) const ->vector_type::const_iterator { return close_begin() + time_to_index(_DaytimeGroup); }

		auto close_it(const size_t& _Idx) const ->vector_type::const_iterator { return close_begin() + _Idx; }


	private:

		auto _index_to_time(size_t _Idx) const ->size_t { return financials::bar_get_unixtime(_Bars.at(_Idx)); }

		auto _index_to_close(size_t _Idx) const ->value_type { return financials::bar_get_close(_Bars.at(_Idx)); }

		void _extract_close_price(vector_type& _Out) const
		{
			std::transform(_Bars.cbegin(), _Bars.cend(), _Out.begin(), [](const bar& _B) {return bar_get_close(_B); });
		}



		financials::bar_vector		_Bars;
		vector_type					_Closes;
		bool						_Good;
	};

	inline bool _Failure(financials::data& _Data) 
	{
		if (!_Data.size()) 
		{
			cout << MSGERRORDATA << "\n";
			cout << MSGERROREXIT << "\n";
			return true;
		}

		return false;
	}

}
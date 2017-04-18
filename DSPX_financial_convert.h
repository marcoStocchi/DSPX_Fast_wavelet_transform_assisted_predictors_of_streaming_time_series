// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once


namespace convert
{
	// time format conversions

	template <class char_type>
	struct time_format_t
	{
		typedef std::basic_string<char_type>			string_type;
		typedef std::basic_istringstream<char_type>		istringstream_type;
		typedef std::basic_ostringstream<char_type>		ostringstream_type;
		typedef std::chrono::system_clock::time_point	time_point_type;



		time_format_t(const time_t& _T, const int& _Dst = -1) : _t(_T)
		{
			//gmtime_s(&_tm, &_T); // UTC
			localtime_s(&_tm, &_T);

			_tm.tm_isdst = _Dst;
		}

		time_format_t(const int& _Year, const int& _Mon, const int& _Day,
			const int& _Hour, const int& _Min, const int& _Sec = 0,
			const int& _Dst = -1)
		{
			_tm.tm_sec = _Sec;
			_tm.tm_min = _Min;
			_tm.tm_hour = _Hour;
			_tm.tm_mday = _Day;
			_tm.tm_mon = _Mon - 1;
			_tm.tm_year = _Year - 1900;
			_tm.tm_isdst = _Dst;

			_t = std::mktime(&_tm);
		}

		time_format_t(const string_type& _DaytimeGroup)
		{// eg."2016.11.27 09:40"
			*this = _DaytimeGroup;
		}


		void operator = (const string_type& _DaytimeGroup)
		{// assign new daytime
			istringstream_type iss(_DaytimeGroup);

			iss >> _tm.tm_year; _tm.tm_year -= 1900;
			iss.ignore(1); iss >> _tm.tm_mon; _tm.tm_mon -= 1;
			iss.ignore(1); iss >> _tm.tm_mday;
			iss.ignore(1); iss >> _tm.tm_hour;
			iss.ignore(1); iss >> _tm.tm_min;

			_tm.tm_sec = 0;
			_tm.tm_isdst = -1;

			_t = std::mktime(&_tm);
		}


		// query

		auto year() const ->int { return 1900 + _tm.tm_year; }

		auto month() const ->int { return 1 + _tm.tm_mon; }

		auto day() const ->int { return _tm.tm_mday; }

		auto hours() const ->int { return _tm.tm_hour; }

		auto minutes() const ->int { return _tm.tm_min; }

		auto seconds() const ->int { return _tm.tm_sec; }

		auto week_day() const ->int { return _tm.tm_wday; /*sunday==0*/ }


		// test

		bool operator == (const time_format_t& _TS) const { return _t == _TS._t; }

		bool operator != (const time_format_t& _TS) const { return _t != _TS._t; }


		// conversions

		auto to_datetime() const ->string_type
		{
			ostringstream_type x;

			x << std::setfill(x.widen('0'));
			
			x << year() << "."
				<< std::setw(2) << month() << "."
				<< std::setw(2) << day() << " "
				<< std::setw(2) << hours() << ":"
				<< std::setw(2) << minutes();

			return x.str();
		}

		auto to_date() const ->string_type
		{
			ostringstream_type x;

			x << std::setfill(x.widen('0'));

			x << year() << "."
				<< std::setw(2) << month() << "."
				<< std::setw(2) << day();

			return x.str();
		}

		auto to_time() const ->string_type
		{
			ostringstream_type x;

			x << std::setfill(x.widen('0'));

			x << std::setw(2) << hours() << ":"
				<< std::setw(2) << minutes();

			return x.str();
		}

		auto to_unix_time() const ->time_t { return _t; }

		auto to_chrono_time_point() const ->time_point_type { return std::chrono::system_clock::from_time_t(_t); }


	private:

		time_t	_t;
		tm		_tm;
	};

	typedef time_format_t<char>		time_string;
	typedef time_format_t<wchar_t>	time_wstring;
}
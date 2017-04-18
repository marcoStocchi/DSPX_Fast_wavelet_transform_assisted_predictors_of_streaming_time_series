// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

namespace financials
{
	// bar type definition
	typedef std::tuple<size_t, double, double, double, double>		bar;
	typedef std::vector<bar>										bar_vector;

	inline size_t bar_get_unixtime(const bar& b) { return std::get<0>(b); }

	inline double bar_get_open(const bar& b) { return std::get<1>(b); }

	inline double bar_get_high(const bar& b) { return std::get<2>(b); }

	inline double bar_get_low(const bar& b) { return std::get<3>(b); }

	inline double bar_get_close(const bar& b) { return std::get<4>(b); }


	// output
	inline std::ostream& operator << (std::ostream& s, const bar& b)
	{
		s << std::get<0>(b) << " " <<
			std::get<1>(b) << " " <<
			std::get<2>(b) << " " <<
			std::get<3>(b) << " " <<
			std::get<4>(b);
		return s;
	}

	// load bar
	inline bar load_bar(std::istream& in)
	{
		size_t t(0); in >> t;

		in.ignore(18); // drop string time eg."2013.01.01 00:00"

		double v[4] ={ 0 };
		for (size_t i=0; i<4; ++i) { in >> v[i]; in.ignore(1); }

		return bar(t, v[0], v[1], v[2], v[3]);
	}

	// load bars, push it in a vector
	inline bool load_bars(const path_type& p, std::vector<bar>& v)
	{
		std::ifstream fin(p.string());

		if (!fin.good()) return false;

		while (fin.good())
		{
			v.push_back(load_bar(fin));
		}

		v.pop_back();

		return true;
	}
}
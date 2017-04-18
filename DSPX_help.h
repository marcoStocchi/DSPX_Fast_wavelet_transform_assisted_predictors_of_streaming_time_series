// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once


namespace predictor_system
{
	// streams

	inline void set_stream(std::ostream& _S) {_S << std::fixed << std::setprecision(8);}
	
	
	// mem

	template <class T> inline void safe_delete(T*& ptr) { if (ptr) { delete ptr; ptr=0; } }

	template <class T> inline void safe_delete(std::vector<T*>& _Cont) 
	{ 
		for (auto I=_Cont.begin(), E=_Cont.end(); I!=E; ++I)
		
			if (*I) { delete *I; *I=0; } 
	}

}
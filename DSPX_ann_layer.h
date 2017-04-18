// Copyright (c) <2016> <Marco Stocchi, UNICA>
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

namespace artificial_neural_networks
{
	template <class C>
	class layer
	{// generic artificial neural networks layer
		// holds the neurons container
		// provides iterator access and basic bulk actions to contained neurons
		// provides protected functions for derived classes

		typedef C									container_type;

	public:

		typedef typename C::value_type				neuron_type;
		typedef typename C::const_iterator			const_iterator;
		typedef typename C::iterator				iterator;
		typedef typename C::const_reverse_iterator	const_reverse_iterator;
		typedef typename C::reverse_iterator		reverse_iterator;


		layer(size_t sz)
			: _neurons(sz) // init container with sz default elements
		{}

		~layer() {}


		auto cbegin() const -> const_iterator { return _neurons.begin(); }

		auto begin() const -> const_iterator { return _neurons.begin(); }

		auto begin() -> iterator { return _neurons.begin(); }


		auto cend() const -> const_iterator { return _neurons.end(); }

		auto end() const -> const_iterator { return _neurons.end(); }

		auto end() -> iterator { return _neurons.end(); }


		auto crbegin() const ->const_reverse_iterator { return _neurons.rbegin(); }

		auto rbegin() const ->const_reverse_iterator { return _neurons.rbegin(); }

		auto rbegin() ->reverse_iterator { return _neurons.rbegin(); }


		auto crend() const ->const_reverse_iterator { return _neurons.crend(); }

		auto rend() const ->const_reverse_iterator { return _neurons.rend(); }

		auto rend() ->reverse_iterator { return _neurons.rend(); }


		auto size() const ->size_t { return _neurons.size(); }

		void truncate() { _neurons.resize(size()-1); }

		virtual void clear() { _neurons.clear(); }

		virtual void erase(iterator& I) { I=_neurons.erase(I); }

		void train_freeze() 
		{ 
			for (auto I=begin(), E=end(); I!=E; ++I) I->weights_freeze();
		}

		void train_revert() 
		{
			for (auto I=begin(), E=end(); I!=E; ++I) I->weights_revert();
		}

		void reduce_learning_rate()
		{
			for (auto I= begin(), E= end(); I!=E; ++I)
				I->reduce_learning_rate();
		}

		void increase_learning_rate()
		{
			for (auto I= begin(), E= end(); I!=E; ++I)
				I->increase_learning_rate();
		}

		void set_learning_rate(const real_type& LR)
		{
			for (auto I= begin(), E= end(); I!=E; ++I)
				I->set_learning_rate(LR);
		}

		auto get_learning_rate() const ->real_type { return cbegin()->get_learning_rate(); }

		void dump_weights(std::ostream& s, char _Endl='\n') const
		{
			for (auto I=cbegin(), E=cend(); I!=E; ++I)

				I->dump_weights(s);
		}

		void save(std::ofstream& fout) const
		{
			for (auto I=cbegin(), E=cend(); I!=E; ++I)

				I->binary_write(fout);
		}

		void dump(size_t _LayerNo, std::ofstream& fout) const
		{
			fout << "LAYER " << _LayerNo << "\n";

			for (size_t i=0; i< _neurons.size(); ++i)

				_neurons.at(i).save(i, fout);

			fout << "\n";
		}

		bool load(std::ifstream& fin, const size_t& _Nsz, const size_t& _Inputsz)
		{
			clear(); _neurons.resize(_Nsz);

			bool success(true);

			for (auto I= begin(), E=end(); I!=E; ++I)

				success = I->binary_read(fin, _Inputsz);

			return success;
		}

	protected:

		typedef layer<C>					layer_type;

		template <class func>
		void _foreach(func&& f)
		{
			std::for_each(begin(), end(), f);
		}

		template <class func>
		void _foreach(func&& f) const
		{
			std::for_each(cbegin(), cend(), f) :
		}

		void _initialize(const size_t& _Sz)
		{
			_foreach(std::bind2nd(std::mem_fun_ref(&neuron_type::initialize), _Sz));
		}

		void _push_back_initialize(const size_t& _Sz)
		{
			neuron_type N; _neurons.push_back(N);
			
			_neurons.back().initialize(_Sz);
		}

		void _revert()
		{
			_foreach(std::mem_fun_ref(&neuron_type::weights_revert));
		}

		void _erase(const_iterator EE)
		{
			_neurons.erase(EE);
		}

	private:

		container_type	_neurons;
	};


}
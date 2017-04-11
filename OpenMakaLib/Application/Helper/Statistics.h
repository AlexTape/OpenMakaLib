#ifndef OPENMAKAENGINE_STATISTICS_H
#define OPENMAKAENGINE_STATISTICS_H

#include <map>

using namespace std;

namespace om
{
	class Statistics
	{
	public:

		Statistics();

		virtual ~Statistics();

		void add(string key, string value);

		void write(string filename);

		void reset();

		void setDefaults();

	private:

		map<string, string> stats;
		bool wroteHeader;

		static const string values[];
	};
}

#endif

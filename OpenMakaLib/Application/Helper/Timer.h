#ifndef OPENMAKAENGINE_TIMER_H
#define OPENMAKAENGINE_TIMER_H

#include <time.h>

namespace om
{
	class Timer
	{
	public:

		Timer();

		virtual ~Timer();

		double getMillis() const;

		void restart();

	private:

		clock_t time;
	};
}

#endif //OPENMAKAENGINE_TIMER_H

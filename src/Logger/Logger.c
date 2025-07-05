#include "Logger.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <pthread.h>
#define _XOPEN_SOURCE  
#include <time.h>

static Logger *gLogData = NULL;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER; 

static void fallback_output(const char *msg) {
  write(gLogData->fd, msg, strlen(msg));
}

static struct s_log_struct *log_init(int fd) {
  if (gLogData)
    return gLogData;

  gLogData = malloc(sizeof(struct s_log_struct));
  if (!gLogData)
    return NULL;

  gLogData->enabled = 1;
  gLogData->fd = fd;
  for (size_t i = 0; i < LOG_SUBS_COUNT; i++) {
    gLogData->subs_levels[i] = DEBUG;
  }

  gLogData->output_func = fallback_output;
  return gLogData;
}

int logger_deinit() {
  pthread_mutex_lock(&lock);
  if (gLogData) {
    free(gLogData);
    gLogData = NULL;
    return 1;
  }

  return 0;
  pthread_mutex_unlock(&lock);
}

int logger_init(int fd) {
  pthread_mutex_lock(&lock);
  int rv = log_init(fd) != NULL ? 0 : -1;
  pthread_mutex_unlock(&lock);
  return rv;
}

Logger *get_logger() {
  return gLogData;
}

void log_set_output_level(eLogSubSystem system, eLogLevel level) {
  pthread_mutex_lock(&lock);
  if (gLogData && system < LOG_SUBS_COUNT)
    gLogData->subs_levels[system] = level; 
  pthread_mutex_unlock(&lock);
}

void log_global_on() {
  pthread_mutex_lock(&lock);
  if (gLogData)
    gLogData->enabled = 1;
  pthread_mutex_unlock(&lock);
}

void log_global_off() {
  pthread_mutex_lock(&lock);
  if (gLogData)
    gLogData->enabled = 0;
  pthread_mutex_unlock(&lock);
}

void log_set_output(void (*output_func)(const char *msg)) {
  pthread_mutex_lock(&lock);
  if (gLogData && output_func) {
    gLogData->output_func = output_func;
  }
  pthread_mutex_unlock(&lock);
}

static const char *level_to_str(eLogLevel level) {
  switch (level) {
    case INFORMATION: return "INFO";
    case DEBUG:       return "DEBUG";
    case WARNING:     return "WARN";
    case ERROR:       return "ERROR";
    case CRITICAL:    return "CRIT";
    default:          return "UNKNOWN";
  }
}

static const char *subsystem_to_str(eLogSubSystem sys) {
  switch (sys) {
    case COMMUNICATION: return "COMM";
    case DISPLAY:       return "DISP";
    case SYSTEM:        return "SYS";
    case SENSOR:        return "SENS";
    default:            return "UNDEF";
  }
}

static const char *rfc8601_timespec() {
  static char rfc8601[64];  

  struct timespec tv;
  if (clock_gettime(CLOCK_REALTIME, &tv)) {
    return "Could not get time of day";
  }

  struct tm tm;
  if (!gmtime_r(&tv.tv_sec, &tm)) {
    return "Could not convert time";
  }

  int milliseconds = (int)(tv.tv_nsec / 1000000);

  strftime(rfc8601, sizeof(rfc8601), "%Y-%m-%dT%H:%M:%S", &tm);
  snprintf(rfc8601 + strlen(rfc8601), sizeof(rfc8601) - strlen(rfc8601),
      ".%03dZ", milliseconds);

  return rfc8601;
}

void log_msg(eLogSubSystem system, eLogLevel level, const char *msg) {
  pthread_mutex_lock(&lock);
  if (!gLogData)
    return;

  if (!gLogData->enabled)
    return;

  if (level < gLogData->subs_levels[system])
    return;

  char buffer[512];
  snprintf(buffer, sizeof(buffer), "[%s][%s][%s] %s\n",
      rfc8601_timespec(), 
      subsystem_to_str(system), 
      level_to_str(level), msg);
  
  gLogData->output_func(buffer);
  pthread_mutex_unlock(&lock);
}


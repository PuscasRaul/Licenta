// TODO: Add config struct

#ifndef LOGGER_H
#define LOGGER_H

typedef enum {
  INFORMATION,
  DEBUG,
  WARNING, 
  ERROR,
  CRITICAL
} eLogLevel;

#define LOG_SUBS_COUNT 4
typedef enum {
  COMMUNICATION,
  DISPLAY,
  SYSTEM,
  SENSOR
} eLogSubSystem;

typedef struct s_log_struct {
  eLogLevel subs_levels[LOG_SUBS_COUNT];
  int fd; 
  int enabled;

  void (*output_func)(const char *msg); 
} Logger; 

/**
 * Creates a singleton logger instance with:
 * - All subsystems set to DEBUG level
 * - Global logging enabled
 * - Default output to specified file descriptor
 * 
 * @param fd File descriptor for log output 
 * @return 0 on success, -1 on failure 
 * @note Subsequent calls return success without reinitializing
 */
int logger_init(int fd);

/**
 * Frees allocated memory and sets internal pointer to NULL.
 * Does not close the file descriptor.
 * 
 * @return 1 if logger was destroyed, 0 if logger was already NULL
 * @warning All logging functions become no-ops after this call
 */
int logger_deinit();

/**
 * @return Pointer to Logger struct, or NULL if not initialized
 */
Logger *get_logger();

/**
 * @param system Subsystem to configure
 * @param level Minimum log level 
 * @note Function is safe to call with uninitialized logger (no-op)
 */
void log_set_output_level(eLogSubSystem system, eLogLevel level);

/**
 * @note Function is safe to call with uninitialized logger (no-op)
 */
void log_global_on();

/**
 * When disabled, all log() calls become no-ops regardless of level settings.
 */
void log_global_off(); 

/**
 * Allows redirecting log output to custom handlers 
 * @param output_func Function to handle formatted log messages 
 * @note Function is safe to call with uninitialized logger (no-op)
 */
void log_set_output(
    void (*output_func)(const char *msg)
);

/**
 * Message will be logged only if:
 * - Logger is initialized and globally enabled
 * - Message level >= subsystem's configured minimum level
 * 
 * Output format: "[LEVEL][SUBSYSTEM] message\n"
 * 
 * @param system Subsystem generating the message
 * @param level Severity level of the message
 * @param msg Message string to log
 * @note Function is safe to call with uninitialized logger (no-op)
 * @warning Message is truncated if longer than ~450 characters
 */
void log_msg(eLogSubSystem system, eLogLevel level, const char *msg);

#endif // LOGGER_H

#ifndef SHEARCUDA_GLOBAL_H
#define SHEARCUDA_GLOBAL_H

#include <QtCore/qglobal.h>

#if defined(SHEARCUDA_LIBRARY)
#  define SHEARCUDASHARED_EXPORT Q_DECL_EXPORT
#else
#  if defined(SHEARCUDA_STATIC)
#    define SHEARCUDASHARED_EXPORT
#  else
#    define SHEARCUDASHARED_EXPORT Q_DECL_IMPORT
#  endif
#endif

#endif // SHEARCUDA_GLOBAL_H

#ifndef CRICK_INLINE
  #if defined(__GNUC__)
    #define CRICK_INLINE static __inline__
  #elif defined(_MSC_VER)
    #define CRICK_INLINE static __inline
  #elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define CRICK_INLINE static inline
  #else
    #define CRICK_INLINE
  #endif
#endif

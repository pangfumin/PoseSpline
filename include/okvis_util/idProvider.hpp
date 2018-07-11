#ifndef _ID_PROVIDER_H_
#define _ID_PROVIDER_H_

#include <stdint.h>
#include <iostream>
#include <atomic>

// https://stackoverflow.com/questions/1008019/c-singleton-design-pattern

template <typename IdType = uint64_t>
class IdProvider
{
    public:
        static IdProvider& getInstance()
        {
            static IdProvider    instance; // Guaranteed to be destroyed.
                                  // Instantiated on first use.
            return instance;
        }
     
    public:
        IdProvider(IdProvider const&)      = delete;
        void operator=(IdProvider const&)  = delete;

    
        // get a unique new ID.
        static IdType newId() {
        const IdType retVal = ++_id;
        return retVal;
        }

        // Get the last generated ID.
        static IdType currentId() {
        return _id;
        }

     private:
        IdProvider() {}                    // Constructor? (the {} brackets) are needed here.

        IdProvider(IdProvider const&);              // Don't Implement
        void operator=(IdProvider const&); // Don't implement

        std::atomic<IdType> _id(0);
};
#endif
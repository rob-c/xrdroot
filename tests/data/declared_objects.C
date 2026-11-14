// Classes a macro declares, written by ROOT 6.40.04 into declared-objects-6.40.root:
//   root -l -b -q declared_objects.C+   (ACLiC, so each class has a dictionary)
// tests/test_root_declared_objects.py writes the same objects with xrdroot and holds
// the bytes of each key to ROOT's, then reads ROOT's file back into these classes.
#include "TNamed.h"
#include "TFile.h"
#include <vector>
#include <string>
#include <map>
#include <set>
#include <list>

namespace ns { class Deep { public: int x = 5; }; }

class Versioned {
public:
   int v = 7;
   virtual ~Versioned() {}
   ClassDef(Versioned, 3);
};

class Obj : public TObject {
public:
   int o = 9;
   ClassDefOverride(Obj, 4);
};

class Base1 {
public:
   short b1 = 1;
};

class Inner {
public:
   int fI = 1;
   double fD = 2;
};

class Plain {
public:
   bool b = true;
   char c = '2';
   short s = 3;
   int i = 4;
   long l = 5;
   long long ll = 6;
   unsigned char uc = 7;
   unsigned int ui = 9;
   float f = 12;
   double d = 13;
   Double32_t d32 = 1.5;
   Double32_t d32r = 2.5; //[0,20,10]
   Float16_t f16 = 3.5;
   Float16_t f16r = 4.5; //[0,20,10]
   int arr[3];
   int n = 2;
   double *var; //[n]
   int trans = 0; //!
   std::string str = "hello";
   TString tstr = "tee";
   std::vector<int> vi;
   std::vector<std::string> vs;
   std::map<int, double> mid;
   Inner inner;
   std::vector<Inner> vinner;
   Inner *pinner;
   Plain() {
      arr[0] = 1; arr[1] = 2; arr[2] = 3;
      var = new double[2]; var[0] = 7; var[1] = 8;
      vi.push_back(10); vi.push_back(11);
      vs.push_back("ab");
      mid[3] = 4.5;
      vinner.push_back(Inner());
      pinner = new Inner();
   }
};

class Derived : public Base1 {
public:
   ns::Deep deep;
   int arr2[2][3];
   std::vector<std::vector<float> > vvf;
   std::vector<Inner*> vpinner;
   std::vector<Versioned> vver;
   std::map<int, std::string> mis;
   std::set<int> si;
   std::list<double> ld;
   std::vector<bool> vb;
   Obj obj;
   Obj *pobj;
   Obj *pnull;
   Versioned ver;
   Derived() {
      for (int i = 0; i < 2; ++i) for (int j = 0; j < 3; ++j) arr2[i][j] = i * 3 + j;
      std::vector<float> one; one.push_back(1.5);
      std::vector<float> two; two.push_back(2.5); two.push_back(3.5);
      vvf.push_back(one); vvf.push_back(two);
      vpinner.push_back(new Inner()); vpinner.push_back(nullptr);
      vver.push_back(Versioned());
      mis[1] = "one"; mis[2] = "two";
      si.insert(4); si.insert(5);
      ld.push_back(0.25);
      vb.push_back(true); vb.push_back(false);
      pobj = new Obj();
      pnull = nullptr;
   }
};

class Named : public TNamed {
public:
   int fN = 0;
   Named() {}
   Named(const char *name, const char *title) : TNamed(name, title), fN(3) {}
   ClassDefOverride(Named, 2);
};

void declared_objects(const char *file = "declared-objects-6.40.root") {
   TFile f(file, "RECREATE");
   Plain plain;
   f.WriteObject(&plain, "plain");
   Derived derived;
   f.WriteObject(&derived, "derived");
   Named named("a", "b");
   named.Write();
   f.Close();
}

// What ROOT 6.40.04 writes for trees of objects: the donor of object-branches-6.40.root.
//
//    root -b -q -l object_branches.C
//
// Five entries of each kind of branch xrdroot writes objects into: a std::vector of
// numbers, a macro's class split member by member (packed floats among them), a
// ROOT::Math four-vector split, a vector of them split as a collection, a
// TLorentzVector streamed whole, histograms whole (TH1F as a TBranchElement, TH2F as
// a TBranchObject) and a TClonesArray of TLines written member-wise.
#include "Math/Vector4D.h"
#include "TClonesArray.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLine.h"
#include "TLorentzVector.h"
#include "TTree.h"
#include <vector>

class Packed {
public:
   Double_t fD;   // a double
   Float_t fF;    // a float
   Int_t fI;      // an int
   Double32_t fP; //[-10,10,12] a packed double
   Float16_t fH;  //[0,0,8] a packed float
};

void object_branches()
{
   TFile f("object-branches-6.40.root", "RECREATE");
   TTree t("t", "objects");
   std::vector<float> numbers, *pn = &numbers;
   auto packed = new Packed;
   auto lv = new ROOT::Math::XYZTVector;
   std::vector<ROOT::Math::XYZTVector> lvs, *plvs = &lvs;
   auto tlv = new TLorentzVector;
   TLorentzVector::Class()->IgnoreTObjectStreamer();
   TVector3::Class()->IgnoreTObjectStreamer();
   auto h1 = new TH1F("h1", "one", 4, 0, 4);
   auto h2 = new TH2F("h2", "two", 2, 0, 2, 2, 0, 2);
   auto lines = new TClonesArray("TLine");
   t.Branch("numbers", &pn);
   t.Branch("packed", &packed, 32000, 1);
   t.Branch("lv", "ROOT::Math::XYZTVector", &lv);
   t.Branch("lvs", "std::vector<ROOT::Math::XYZTVector>", &plvs);
   t.Branch("tlv", "TLorentzVector", &tlv, 32000, 2);
   t.Branch("h1", "TH1F", &h1, 32000, 0);
   t.Branch("h2", "TH2F", &h2, 32000, 0);
   t.Branch("lines", &lines, 32000, 0);
   lines->BypassStreamer();
   for (int i = 0; i < 5; ++i) {
      numbers.assign(i, 0.5f * i);
      packed->fD = packed->fF = packed->fI = packed->fP = packed->fH = 1.25 * i - 2;
      lv->SetCoordinates(i, 2 * i, 3 * i, 10 + i);
      lvs.assign(i % 3, *lv);
      tlv->SetPxPyPzE(i, -i, 2 * i, 20 + i);
      h1->Fill(i);
      h2->Fill(i % 2, i / 3);
      lines->Clear();
      for (int k = 0; k < i; ++k)
         new ((*lines)[k]) TLine(k, i, k + 1, i + 0.5);
      t.Fill();
   }
   t.Write();
}


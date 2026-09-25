// Written by ROOT 6.40.04, run interpreted: root -b -q tclonesarray_split.C
// Write tclonesarray-split.root: a TClonesArray of Hit split into members,
// and one branch whose baskets ROOT is told to put in tclonesarray-split-baskets.root.
#include "TClonesArray.h"
#include "TFile.h"
#include "TNamed.h"
#include "TTree.h"

class Hit : public TObject {
public:
   Int_t fId = 0;
   Double32_t fE = 0; //[0,100,16]
   Float_t fPos[3] = {0, 0, 0};
   TString fLabel;
   ClassDefOverride(Hit, 1)
};

void tclonesarray_split()
{
   TFile f("tclonesarray-split.root", "RECREATE");
   TTree t("t", "hits split into members");
   TClonesArray hits("Hit");
   Int_t n = 0;
   t.Branch("hits", &hits, 32000, 99);
   t.Branch("n", &n, "n/I");
   TFile far("tclonesarray-split-baskets.root", "RECREATE");
   f.cd();
   t.GetBranch("n")->SetFile(&far);
   for (Int_t entry = 0; entry < 5; ++entry) {
      hits.Clear();
      n = entry;
      for (Int_t i = 0; i < entry; ++i) {
         Hit *hit = new (hits[i]) Hit;
         hit->fId = 10 * entry + i;
         hit->fE = 12.5 * i + entry;
         hit->fPos[0] = entry;
         hit->fPos[1] = i;
         hit->fPos[2] = -1;
         hit->fLabel = TString::Format("hit-%d-%d", entry, i);
      }
      t.Fill();
   }
   t.FlushBaskets();
   f.cd();
   t.Write();
   f.Close();
   far.Close();
}

// Written by ROOT 6.40.04, run interpreted: root -b -q tail_basket.C
// Write tail-basket.root: a tree whose branch has flushed three baskets and
// is then written with its fourth still being filled, as WriteTObject - which
// is not TTree::Write, and does not FlushBaskets first - leaves it.
#include "TFile.h"
#include "TTree.h"

void tail_basket()
{
   TFile f("tail-basket.root", "RECREATE");
   TTree t("t", "a basket still being filled");
   Int_t i = 0;
   t.Branch("i", &i, "i/I");
   t.SetBasketSize("i", 64);
   for (i = 0; i < 25; ++i)
      t.Fill();
   f.WriteTObject(&t);
   f.Close();
}

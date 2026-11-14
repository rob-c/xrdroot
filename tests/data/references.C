// Writes references-6.40.root: an object a TRef points at, the TRef, a TRefArray of it
// and of a bare TObject, the TObject, and a TBits - with ROOT 6.40's ProcessID0 record
// beside them. Run with: root -b -q references.C
void references()
{
   TFile *f = TFile::Open("references-6.40.root", "RECREATE");
   TNamed *obj = new TNamed("target", "a referenced object");
   TRef ref = obj;
   TRefArray arr;
   arr.Add(obj);
   TObject *other = new TObject();
   arr.Add(other);
   TBits bits(5);
   bits.SetBitNumber(1);
   bits.SetBitNumber(17);
   f->WriteObject(obj, "therefobj");
   f->WriteObject(&ref, "theref");
   f->WriteObject(&arr, "thearray");
   f->WriteObject(other, "other");
   f->WriteObject(&bits, "bits");
   delete f;
}

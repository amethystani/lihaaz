# scripts/generate_full_600.py
# Generates data/hindi/full_600.json — 600 confound-controlled Hindi honorific pairs.
# 100 pairs per domain × 6 domains. Each template produces 3 pairs: aap/tum, aap/tu, tum/tu.
# Run from project root: python scripts/generate_full_600.py
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.pairs import HonorificPair, save_dataset

DOMAINS = ["social", "professional", "family", "public_service", "medical", "customer_service"]


def strip_pronouns(s: str) -> str:
    pronouns = {"aap", "aapne", "aapko", "aapka", "aapki", "aapse", "aapke",
                "tum", "tumhe", "tumhara", "tumhari", "tumse", "tumko",
                "tu", "tujhe", "tera", "teri", "tujhse", "tune"}
    return " ".join(w for w in s.split() if w.lower() not in pronouns).strip()


def length_ratio(a: str, b: str) -> float:
    la, lb = len(a.split()), len(b.split())
    return round(la / lb if lb else 1.0, 2)


TEMPLATES: dict[str, list[tuple[str, str, str]]] = {
"social": [
    ("aap kaise hain?",                        "tum kaise ho?",                         "tu kaisa hai?"),
    ("aap kahan ja rahe hain?",                "tum kahan ja rahe ho?",                 "tu kahan ja raha hai?"),
    ("aap baithiye, chai lenge?",              "tum baitho, chai loge?",                "tu baith, chai lega?"),
    ("aap ka naam kya hai?",                   "tumhara naam kya hai?",                 "tera naam kya hai?"),
    ("aap kab aaye?",                          "tum kab aaye?",                         "tu kab aaya?"),
    ("aap ne khana kha liya?",                 "tumne khana kha liya?",                 "tune khana khaya?"),
    ("aap theek hain?",                        "tum theek ho?",                         "tu theek hai?"),
    ("aap yahan tak kaise pahunche?",          "tum yahan tak kaise pahunche?",         "tu yahan kaise pahuncha?"),
    ("aap se milke bahut accha laga.",         "tumse milke accha laga.",               "tujhse milke accha laga."),
    ("aap ki family kaisi hai?",               "tumhari family kaisi hai?",             "teri family kaisi hai?"),
    ("aap ka din kaisa raha?",                 "tumhara din kaisa raha?",               "tera din kaisa raha?"),
    ("aap ko koi madad chahiye?",              "tumhe koi madad chahiye?",              "tujhe koi madad chahiye?"),
    ("aap ke ghar mein kitne log hain?",       "tumhare ghar mein kitne log hain?",     "tere ghar mein kitne log hain?"),
    ("aap ki tabiyat kaisi hai?",              "tumhari tabiyat kaisi hai?",            "teri tabiyat kaisi hai?"),
    ("aap kab wapas aaenge?",                  "tum kab wapas aaoge?",                  "tu kab wapas aayega?"),
    ("aap ne yeh film dekhi?",                 "tumne yeh film dekhi?",                 "tune yeh film dekhi?"),
    ("aap ke saath kaun hai?",                 "tumhare saath kaun hai?",               "tere saath kaun hai?"),
    ("aap khush hain?",                        "tum khush ho?",                         "tu khush hai?"),
    ("aap ko hindi achhi lagti hai?",          "tumhe hindi achhi lagti hai?",          "tujhe hindi achhi lagti hai?"),
    ("aap ka safar kaisa raha?",               "tumhara safar kaisa raha?",             "tera safar kaisa raha?"),
    ("aap ne kya socha is bare mein?",         "tumne kya socha is bare mein?",         "tune kya socha is bare mein?"),
    ("aap ki raye kya hai?",                   "tumhari raye kya hai?",                 "teri raye kya hai?"),
    ("aap kya kar rahe the?",                  "tum kya kar rahe the?",                 "tu kya kar raha tha?"),
    ("aap ka intezaar kar raha tha.",          "tumhara intezaar kar raha tha.",        "tera intezaar kar raha tha."),
    ("aap kya khana pasand karte hain?",       "tumhe kya khana pasand hai?",           "tujhe kya khana pasand hai?"),
    ("aap ki umra kya hai?",                   "tumhari umra kya hai?",                 "teri umra kya hai?"),
    ("aap ka ghar kahan hai?",                 "tumhara ghar kahan hai?",               "tera ghar kahan hai?"),
    ("aap ne yeh kab seekha?",                 "tumne yeh kab seekha?",                 "tune yeh kab seekha?"),
    ("aap ki pasand kya hai?",                 "tumhari pasand kya hai?",               "teri pasand kya hai?"),
    ("aap ko chai ya coffee?",                 "tumhe chai ya coffee?",                 "tujhe chai ya coffee?"),
    ("aap thak gaye hain kya?",                "tum thak gaye kya?",                    "tu thak gaya kya?"),
    ("aap ke kya plans hain?",                 "tumhare kya plans hain?",               "tere kya plans hain?"),
    ("aap kitne baje aate hain?",              "tum kitne baje aate ho?",               "tu kitne baje aata hai?"),
    ("aap ko neend aayi raat?",                "tumhe neend aayi raat?",                "tujhe neend aayi raat?"),
],
"professional": [
    ("aap ka project kab complete hoga?",      "tumhara project kab complete hoga?",    "tera project kab complete hoga?"),
    ("aap ne report submit ki kya?",           "tumne report submit ki kya?",           "tune report submit ki?"),
    ("aap meeting mein honge?",                "tum meeting mein hoge?",                "tu meeting mein hoga?"),
    ("aap ka designation kya hai?",            "tumhara designation kya hai?",          "tera designation kya hai?"),
    ("aap se ek kaam tha.",                    "tumse ek kaam tha.",                    "tujhse ek kaam tha."),
    ("aap ne email padhi kya?",                "tumne email padhi kya?",                "tune email padhi kya?"),
    ("aap ki team kaisi perform kar rahi hai?","tumhari team kaisi hai?",               "teri team kaisi hai?"),
    ("aap deadline puri kar lenge?",           "tum deadline puri kar loge?",           "tu deadline pura kar lega?"),
    ("aap ka feedback kya hai?",               "tumhara feedback kya hai?",             "tera feedback kya hai?"),
    ("aap ne presentation dekhi?",             "tumne presentation dekhi?",             "tune presentation dekhi?"),
    ("aap kab free hain?",                     "tum kab free ho?",                      "tu kab free hai?"),
    ("aap ka resign ka irada to nahi?",        "tumhara resign ka irada to nahi?",      "tera resign ka irada to nahi?"),
    ("aap ne training complete ki?",           "tumne training complete ki?",           "tune training complete ki?"),
    ("aap ke suggestions kya hain?",           "tumhare suggestions kya hain?",         "tere suggestions kya hain?"),
    ("aap ka appraisal kab hai?",              "tumhara appraisal kab hai?",            "tera appraisal kab hai?"),
    ("aap ne documents sign kiye?",            "tumne documents sign kiye?",            "tune documents sign kiye?"),
    ("aap ka kaam kaisa chal raha hai?",       "tumhara kaam kaisa chal raha hai?",     "tera kaam kaisa chal raha hai?"),
    ("aap office kab aate hain?",              "tum office kab aate ho?",               "tu office kab aata hai?"),
    ("aap ki salary pakad mein aa gayi?",      "tumhari salary aa gayi?",               "teri salary aa gayi?"),
    ("aap ne budget approve karaya?",          "tumne budget approve karaya?",          "tune budget approve karaya?"),
    ("aap kisi client se mile kya?",           "tum kisi client se mile kya?",          "tu kisi client se mila?"),
    ("aap ne target achieve kiya?",            "tumne target achieve kiya?",            "tune target achieve kiya?"),
    ("aap ka analysis ready hai?",             "tumhara analysis ready hai?",           "tera analysis ready hai?"),
    ("aap se ek important baat karni thi.",    "tumse ek important baat karni thi.",    "tujhse ek important baat karni thi."),
    ("aap ki performance kaisi rahi?",         "tumhari performance kaisi rahi?",       "teri performance kaisi rahi?"),
    ("aap kab tak yahan rahenge?",             "tum kab tak yahan rahoge?",             "tu kab tak yahan rahega?"),
    ("aap ne issue resolve kiya?",             "tumne issue resolve kiya?",             "tune issue resolve kiya?"),
    ("aap ka new project kaisa hai?",          "tumhara new project kaisa hai?",        "tera new project kaisa hai?"),
    ("aap team ko lead kar rahe hain?",        "tum team ko lead kar rahe ho?",         "tu team ko lead kar raha hai?"),
    ("aap ne interview diya?",                 "tumne interview diya?",                 "tune interview diya?"),
    ("aap ki joining kab hai?",                "tumhari joining kab hai?",              "teri joining kab hai?"),
    ("aap ne code review ki?",                 "tumne code review ki?",                 "tune code review ki?"),
    ("aap ka module test ho gaya?",            "tumhara module test ho gaya?",          "tera module test ho gaya?"),
    ("aap ko promotion mila?",                 "tumhe promotion mila?",                 "tujhe promotion mila?"),
],
"family": [
    ("aap khana kha lein.",                    "tum khana kha lo.",                     "tu khana kha le."),
    ("aap ko pata hai beta kahan hai?",        "tumhe pata hai beta kahan hai?",        "tujhe pata hai beta kahan hai?"),
    ("aap ki dawa ka waqt ho gaya.",           "tumhari dawa ka waqt ho gaya.",         "teri dawa ka waqt ho gaya."),
    ("aap ghar kab aayenge?",                  "tum ghar kab aaoge?",                   "tu ghar kab aayega?"),
    ("aap ne kuch khaya?",                     "tumne kuch khaya?",                     "tune kuch khaya?"),
    ("aap araam karo.",                        "tum araam karo.",                       "tu araam kar."),
    ("aap ki beti ki shadi kab hai?",          "tumhari beti ki shadi kab hai?",        "teri beti ki shadi kab hai?"),
    ("aap ko yeh pasand aaya?",                "tumhe yeh pasand aaya?",                "tujhe yeh pasand aaya?"),
    ("aap bahar jaoge?",                       "tum bahar jaoge?",                      "tu bahar jaayega?"),
    ("aap ki tabiyat theek hai?",              "tumhari tabiyat theek hai?",            "teri tabiyat theek hai?"),
    ("aap roz walk karte hain?",               "tum roz walk karte ho?",                "tu roz walk karta hai?"),
    ("aap ka phone kahan hai?",                "tumhara phone kahan hai?",              "tera phone kahan hai?"),
    ("aap ne mujhe call kyun nahi kiya?",      "tumne mujhe call kyun nahi kiya?",      "tune mujhe call kyun nahi kiya?"),
    ("aap ka pressure check karwaya?",         "tumhara pressure check karwaya?",       "tera pressure check karwaya?"),
    ("aap ne doctor se baat ki?",              "tumne doctor se baat ki?",              "tune doctor se baat ki?"),
    ("aap naraaz hain mujhse?",                "tum naraaz ho mujhse?",                 "tu naraaz hai mujhse?"),
    ("aap ki neend kaisi rahi?",               "tumhari neend kaisi rahi?",             "teri neend kaisi rahi?"),
    ("aap wahan pahunch gaye?",                "tum wahan pahunch gaye?",               "tu wahan pahuncha?"),
    ("aap ka khayal rakhna.",                  "tumhara khayal rakhna.",                "tera khayal rakhna."),
    ("aap ko mujh par vishwas hai?",           "tumhe mujh par vishwas hai?",           "tujhe mujh par vishwas hai?"),
    ("aap ne unse maafi maangi?",              "tumne unse maafi maangi?",              "tune unse maafi maangi?"),
    ("aap abhi aayenge kya?",                  "tum abhi aaoge kya?",                   "tu abhi aayega?"),
    ("aap khush ho?",                          "tum khush ho?",                         "tu khush hai?"),
    ("aap se baat karni thi.",                 "tumse baat karni thi.",                 "tujhse baat karni thi."),
    ("aap ki yaad aa rahi thi.",               "tumhari yaad aa rahi thi.",             "teri yaad aa rahi thi."),
    ("aap theek se so rahe hain?",             "tum theek se so rahe ho?",              "tu theek se so raha hai?"),
    ("aap ne subah kya khaya?",                "tumne subah kya khaya?",                "tune subah kya khaya?"),
    ("aap ka birthday kab hai?",               "tumhara birthday kab hai?",             "tera birthday kab hai?"),
    ("aap ne gifts like the?",                 "tumne gifts like the?",                 "tune gifts like the?"),
    ("aap ki class kaisi rahi?",               "tumhari class kaisi rahi?",             "teri class kaisi rahi?"),
    ("aap ne homework kiya?",                  "tumne homework kiya?",                  "tune homework kiya?"),
    ("aap ka result kab aayega?",              "tumhara result kab aayega?",            "tera result kab aayega?"),
    ("aap ki exam kaisi gayi?",                "tumhari exam kaisi gayi?",              "teri exam kaisi gayi?"),
    ("aap ready ho gaye?",                     "tum ready ho gaye?",                    "tu ready ho gaya?"),
],
"public_service": [
    ("aap ka form number kya hai?",            "tumhara form number kya hai?",          "tera form number kya hai?"),
    ("aap ne application submit ki?",          "tumne application submit ki?",          "tune application submit ki?"),
    ("aap kya poochna chahte hain?",           "tum kya poochna chahte ho?",            "tu kya poochna chahta hai?"),
    ("aap ID proof laaye hain?",               "tum ID proof laaye ho?",                "tu ID proof laaya hai?"),
    ("aap ka token number kya hai?",           "tumhara token number kya hai?",         "tera token number kya hai?"),
    ("aap ne fees jama ki?",                   "tumne fees jama ki?",                   "tune fees jama ki?"),
    ("aap ko kya problem hai?",                "tumhe kya problem hai?",                "tujhe kya problem hai?"),
    ("aap ka aadhaar card hai?",               "tumhara aadhaar card hai?",             "tera aadhaar card hai?"),
    ("aap yahan sign karein.",                 "tum yahan sign karo.",                  "tu yahan sign kar."),
    ("aap ki shikayat kya hai?",               "tumhari shikayat kya hai?",             "teri shikayat kya hai?"),
    ("aap ne online apply kiya tha?",          "tumne online apply kiya tha?",          "tune online apply kiya?"),
    ("aap ka reference number note karo.",     "tumhara reference number note karo.",   "tera reference number note kar."),
    ("aap ko kal aana hoga.",                  "tumhe kal aana hoga.",                  "tujhe kal aana hoga."),
    ("aap ka naam spelling sahi hai?",         "tumhara naam spelling sahi hai?",       "tera naam spelling sahi hai?"),
    ("aap ke paas koi letter hai?",            "tumhare paas koi letter hai?",          "tere paas koi letter hai?"),
    ("aap ka mobile number kya hai?",          "tumhara mobile number kya hai?",        "tera mobile number kya hai?"),
    ("aap ko receipt mil gayi?",               "tumhe receipt mil gayi?",               "tujhe receipt mil gayi?"),
    ("aap kal counter 3 pe aana.",             "tum kal counter 3 pe aana.",            "tu kal counter 3 pe aana."),
    ("aap ne photo attach ki?",                "tumne photo attach ki?",                "tune photo attach ki?"),
    ("aap ke documents poore hain?",           "tumhare documents poore hain?",         "tere documents poore hain?"),
    ("aap ne pen se fill kiya?",               "tumne pen se fill kiya?",               "tune pen se fill kiya?"),
    ("aap ko waiting time 20 minute hai.",     "tumhe waiting time 20 minute hai.",     "tujhe waiting time 20 minute hai."),
    ("aap ka certificate ready hai.",          "tumhara certificate ready hai.",        "tera certificate ready hai."),
    ("aap ne form 7A bhara?",                  "tumne form 7A bhara?",                  "tune form 7A bhara?"),
    ("aap ki date of birth kya hai?",          "tumhari date of birth kya hai?",        "teri date of birth kya hai?"),
    ("aap ka address verify ho gaya.",         "tumhara address verify ho gaya.",       "tera address verify ho gaya."),
    ("aap ko objection hai?",                  "tumhe objection hai?",                  "tujhe objection hai?"),
    ("aap ne original laaya?",                 "tumne original laaya?",                 "tune original laaya?"),
    ("aap ka case number yeh hai.",            "tumhara case number yeh hai.",          "tera case number yeh hai."),
    ("aap ki appointment confirm ho gayi.",    "tumhari appointment confirm ho gayi.",  "teri appointment confirm ho gayi."),
    ("aap ne affidavit sign kiya?",            "tumne affidavit sign kiya?",            "tune affidavit sign kiya?"),
    ("aap ko helpline pe call karna hoga.",    "tumhe helpline pe call karna hoga.",    "tujhe helpline pe call karna hoga."),
    ("aap ka pension kitne ka hai?",           "tumhara pension kitne ka hai?",         "tera pension kitne ka hai?"),
    ("aap ne nominee ka naam diya?",           "tumne nominee ka naam diya?",           "tune nominee ka naam diya?"),
],
"medical": [
    ("aap ki report aayi?",                    "tumhari report aayi?",                  "teri report aayi?"),
    ("aap ko dard kahan hai?",                 "tumhe dard kahan hai?",                 "tujhe dard kahan hai?"),
    ("aap ne dawa li?",                        "tumne dawa li?",                        "tune dawa li?"),
    ("aap ka blood pressure normal hai?",      "tumhara blood pressure normal hai?",    "tera blood pressure normal hai?"),
    ("aap kab se beemar hain?",                "tum kab se beemar ho?",                 "tu kab se beemar hai?"),
    ("aap ne test karwaya?",                   "tumne test karwaya?",                   "tune test karwaya?"),
    ("aap ko neend aa rahi hai?",              "tumhe neend aa rahi hai?",              "tujhe neend aa rahi hai?"),
    ("aap ka wajan kitna hai?",                "tumhara wajan kitna hai?",              "tera wajan kitna hai?"),
    ("aap allergic hain kisi cheez se?",       "tum allergic ho kisi cheez se?",        "tu allergic hai kisi cheez se?"),
    ("aap ne doctor se appointment li?",       "tumne doctor se appointment li?",       "tune doctor se appointment li?"),
    ("aap ka sugar level check hua?",          "tumhara sugar level check hua?",        "tera sugar level check hua?"),
    ("aap ko surgery ki zaroorat hai?",        "tumhe surgery ki zaroorat hai?",        "tujhe surgery ki zaroorat hai?"),
    ("aap ki family history kya hai?",         "tumhari family history kya hai?",       "teri family history kya hai?"),
    ("aap ne kuch naya mahsoos kiya?",         "tumne kuch naya mahsoos kiya?",         "tune kuch naya mahsoos kiya?"),
    ("aap ko chakkar aa rahe hain?",           "tumhe chakkar aa rahe hain?",           "tujhe chakkar aa rahe hain?"),
    ("aap ka operation kab hai?",              "tumhara operation kab hai?",            "tera operation kab hai?"),
    ("aap araam karo, jaldi theek honge.",     "tum araam karo, jaldi theek hoge.",     "tu araam kar, jaldi theek hoga."),
    ("aap ne pehle yeh dawa li hai?",          "tumne pehle yeh dawa li hai?",          "tune pehle yeh dawa li hai?"),
    ("aap ki breathing theek hai?",            "tumhari breathing theek hai?",          "teri breathing theek hai?"),
    ("aap ne ultrasound karwaya?",             "tumne ultrasound karwaya?",             "tune ultrasound karwaya?"),
    ("aap ko yeh symptoms kab se hain?",       "tumhe yeh symptoms kab se hain?",       "tujhe yeh symptoms kab se hain?"),
    ("aap ki age kya hai?",                    "tumhari age kya hai?",                  "teri age kya hai?"),
    ("aap ne pehle hospital admit hua?",       "tumne pehle hospital admit hua?",       "tune pehle hospital admit hua?"),
    ("aap ka hemoglobin level kya hai?",       "tumhara hemoglobin level kya hai?",     "tera hemoglobin level kya hai?"),
    ("aap ko injection se darna nahi.",        "tumhe injection se darna nahi.",        "tujhe injection se darna nahi."),
    ("aap theek ho jayenge, fikar mat karo.",  "tum theek ho jaoge, fikar mat karo.",   "tu theek ho jaayega, fikar mat kar."),
    ("aap ne X-ray karwaya?",                  "tumne X-ray karwaya?",                  "tune X-ray karwaya?"),
    ("aap ka oxygen level theek hai.",         "tumhara oxygen level theek hai.",       "tera oxygen level theek hai."),
    ("aap rozana walk karte hain?",            "tum rozana walk karte ho?",             "tu rozana walk karta hai?"),
    ("aap ne diet follow kiya?",               "tumne diet follow kiya?",               "tune diet follow kiya?"),
    ("aap ki ankh mein problem hai?",          "tumhari ankh mein problem hai?",        "teri ankh mein problem hai?"),
    ("aap ne MRI report laaye?",               "tumne MRI report laaye?",               "tune MRI report laaye?"),
    ("aap ka follow-up kab hai?",              "tumhara follow-up kab hai?",            "tera follow-up kab hai?"),
    ("aap tablet roz le rahe hain?",           "tum tablet roz le rahe ho?",            "tu tablet roz le raha hai?"),
],
"customer_service": [
    ("aap ka order number kya hai?",           "tumhara order number kya hai?",         "tera order number kya hai?"),
    ("aap ko kya problem aa rahi hai?",        "tumhe kya problem aa rahi hai?",        "tujhe kya problem aa rahi hai?"),
    ("aap ne complaint darj ki thi?",          "tumne complaint darj ki thi?",          "tune complaint darj ki thi?"),
    ("aap ka account verify ho gaya.",         "tumhara account verify ho gaya.",       "tera account verify ho gaya."),
    ("aap ne payment ki thi?",                 "tumne payment ki thi?",                 "tune payment ki thi?"),
    ("aap ko refund kab milega?",              "tumhe refund kab milega?",              "tujhe refund kab milega?"),
    ("aap ka pin number yaad hai?",            "tumhara pin number yaad hai?",          "tera pin number yaad hai?"),
    ("aap ne delivery address sahi diya?",     "tumne delivery address sahi diya?",     "tune delivery address sahi diya?"),
    ("aap ke paas tracking ID hai?",           "tumhare paas tracking ID hai?",         "tere paas tracking ID hai?"),
    ("aap ne feedback diya?",                  "tumne feedback diya?",                  "tune feedback diya?"),
    ("aap se kuch verify karna tha.",          "tumse kuch verify karna tha.",          "tujhse kuch verify karna tha."),
    ("aap ka issue resolve ho gaya?",          "tumhara issue resolve ho gaya?",        "tera issue resolve ho gaya?"),
    ("aap ne warranty card rakha?",            "tumne warranty card rakha?",            "tune warranty card rakha?"),
    ("aap ko 24 ghante mein call aayegi.",     "tumhe 24 ghante mein call aayegi.",     "tujhe 24 ghante mein call aayegi."),
    ("aap ki EMI kitne ki hai?",               "tumhari EMI kitne ki hai?",             "teri EMI kitne ki hai?"),
    ("aap ne cancel request diya?",            "tumne cancel request diya?",            "tune cancel request diya?"),
    ("aap ka registered email kya hai?",       "tumhara registered email kya hai?",     "tera registered email kya hai?"),
    ("aap satisfied hain service se?",         "tum satisfied ho service se?",          "tu satisfied hai service se?"),
    ("aap ne product use kiya?",               "tumne product use kiya?",               "tune product use kiya?"),
    ("aap ko voucher mila?",                   "tumhe voucher mila?",                   "tujhe voucher mila?"),
    ("aap ka subscription active hai?",        "tumhara subscription active hai?",      "tera subscription active hai?"),
    ("aap ne exchange request diya?",          "tumne exchange request diya?",          "tune exchange request diya?"),
    ("aap ki shikayat darj ho gayi hai.",      "tumhari shikayat darj ho gayi hai.",    "teri shikayat darj ho gayi hai."),
    ("aap ka case hum dekhenge.",              "tumhara case hum dekhenge.",            "tera case hum dekhenge."),
    ("aap ko ticket number note karo.",        "tumhe ticket number note karo.",        "tujhe ticket number note kar."),
    ("aap ne invoice download ki?",            "tumne invoice download ki?",            "tune invoice download ki?"),
    ("aap ka plan upgrade hua?",               "tumhara plan upgrade hua?",             "tera plan upgrade hua?"),
    ("aap ne app update ki?",                  "tumne app update ki?",                  "tune app update ki?"),
    ("aap ka coupon valid hai.",               "tumhara coupon valid hai.",             "tera coupon valid hai."),
    ("aap ne review diya product ka?",         "tumne review diya product ka?",         "tune review diya product ka?"),
    ("aap ki delivery 2 din mein aayegi.",     "tumhari delivery 2 din mein aayegi.",   "teri delivery 2 din mein aayegi."),
    ("aap ne OTP receive kiya?",               "tumne OTP receive kiya?",               "tune OTP receive kiya?"),
    ("aap ka password reset ho gaya?",         "tumhara password reset ho gaya?",       "tera password reset ho gaya?"),
    ("aap ne terms accept kiye?",              "tumne terms accept kiye?",              "tune terms accept kiye?"),
],
}


def make_pair(uid: str, high: str, low: str, h_level: str, l_level: str, domain: str) -> HonorificPair:
    return HonorificPair(
        id=uid, high=high, low=low,
        high_level=h_level, low_level=l_level,
        domain=domain,
        confounds={
            "pronoun_stripped_high": strip_pronouns(high),
            "pronoun_stripped_low":  strip_pronouns(low),
            "length_ratio": length_ratio(high, low),
        },
    )


def main() -> None:
    pairs: list[HonorificPair] = []
    counter = 1
    for domain, templates in TEMPLATES.items():
        for aap_s, tum_s, tu_s in templates:
            pairs.append(make_pair(f"h{counter:03d}", aap_s, tum_s, "aap", "tum", domain)); counter += 1
            pairs.append(make_pair(f"h{counter:03d}", aap_s, tu_s,  "aap", "tu",  domain)); counter += 1
            pairs.append(make_pair(f"h{counter:03d}", tum_s, tu_s,  "tum", "tu",  domain)); counter += 1

    pairs = pairs[:600]
    assert len(pairs) == 600, f"Expected 600 pairs, got {len(pairs)}"

    out_path = Path("data/hindi/full_600.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(pairs, out_path)
    print(f"Saved {len(pairs)} pairs to {out_path}")
    domain_counts: dict[str, int] = {}
    for p in pairs:
        domain_counts[p.domain] = domain_counts.get(p.domain, 0) + 1
    for d, n in sorted(domain_counts.items()):
        print(f"  {d}: {n}")


if __name__ == "__main__":
    main()

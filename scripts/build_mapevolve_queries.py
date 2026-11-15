"""Build 100 reviewable cases from manually inspected MapEvolve images.

Run with: python -m scripts.build_mapevolve_queries
Requires Pillow. No APIs or credentials are used. Source selection and labels
are manual; EXIF extraction and neutral image exports are deterministic.
"""
from __future__ import annotations
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps
from jev_router.dataset import validate, Case

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/mapevolve_queries'
DATASET = 'mapevolve_images'

# Indices refer to the sorted inventory stored alongside the answer key.
# Each tuple: source index, building key, difficulty, question, visible cue.
IDENTIFICATIONS = [
(0,'brittain','easy','Which Georgia Tech dining hall is shown in this interior photograph?','Large stained-glass window and timber-truss dining room.'),
(1,'brittain','moderate','Which dining hall contains this long timber-roofed room?','Long vaulted hall, timber roof, and arched window at the far end.'),
(2,'clough','easy','Which building is this glass-fronted Georgia Tech learning facility?','Glass corner facade and projecting roof sunshades.'),
(3,'clough','moderate','Name the campus building behind the autumn trees.','Layered glass facade, roof shades, and brick side wall.'),
(4,'clough','moderate','Which building contains these broad steps and wooden tiered seating?','Clough Commons signature staircase and wood study terraces.'),
(5,'clough','hard','Identify the building containing this window-lined study space, if you can.','Study tables, tall windows and suspended linear lights; interior is less distinctive.'),
(7,'clough','hard','Which campus building contains this study area viewed from above?','Study seating under a dense hanging-light installation; limited exterior context.'),
(8,'clough','hard','Which building contains this shadowed glass-walled atrium?','Tall glass wall, concrete columns, and study terraces.'),
(9,'clough','hard','Identify the building from this high-angle view of its interior steps.','Broad stair and wooden seating alongside the glass wall.'),
(12,'clough','moderate','Name the Georgia Tech building containing this sunlit tiered study hall.','Distinctive stepped wood seating and tall glazed atrium.'),
(13,'kendeda','easy','Which Georgia Tech sustainability building is this?','Large photovoltaic canopy above a low timber-and-glass building.'),
(14,'kendeda','moderate','Identify this low-rise campus building in the evening.','Broad solar canopy and illuminated timber/glass facade.'),
(15,'kendeda','moderate','What is the name of this campus building viewed across the road at dusk?','Wide solar canopy over the low-rise building.'),
(16,'kendeda','hard','Which building is partly hidden behind this landscaped garden?','Solar canopy and low-rise glazed facade are visible through vegetation.'),
(17,'kendeda','moderate','Which building stands behind this planted rooftop garden?','Vegetated roof area and photovoltaic canopy.'),
(18,'kendeda','hard','Identify the building from this timber-framed interior view.','Exposed timber frame, interior stairs and mezzanine.'),
(19,'campanile','hard','What is the name of the slender campus landmark visible through the blurred leaves?','Narrow metallic spire partially framed by out-of-focus foliage.'),
(20,'campanile','easy','Name the tall sculptural landmark in this campus plaza.','Freestanding tapering metal campanile.'),
(36,'campanile','easy','What is the name of this tower-and-fountain landmark?','Tapering metal spire rising from a circular fountain.'),
(37,'campanile','moderate','Identify the central campus landmark beyond the crowd.','Metal spire and circular basin seen from above.'),
(21,'klaus','hard','Which building is partly obscured by these autumn trees?','Glass entrance and masonry are heavily obscured by foliage.'),
(34,'klaus','moderate','Which Georgia Tech computing building is shown beyond the outdoor stairs?','Broad curved glass facade above an outdoor staircase.'),
(35,'klaus','easy','What is the building name above this rainbow-colored staircase?','Klaus Advanced Computing Building lettering over the steps.'),
(22,'mccamish','hard','Which Georgia Tech venue is shown in this close exterior view?','GT logo and partial arena facade; venue name is outside the frame.'),
(23,'mccamish','easy','Which arena is named on this entrance canopy?','McCamish Pavilion lettering on the entrance.'),
(24,'mccamish','easy','Name the building with this curved glass entrance.','McCamish Pavilion entrance lettering and curved facade.'),
(25,'mccamish','moderate','Which Georgia Tech arena has this entrance with the large GT logo?','Large GT logo and curved arena entrance.'),
(32,'carnegie','moderate','Which historic campus building has this columned entrance?','Red-brick facade, pale classical columns and central entrance.'),
(33,'carnegie','moderate','Identify this red-brick building with the classical portico.','Carnegie facade and central columned portico.'),
(38,'scheller-college','easy','Which college building is named over these entrance doors?','Scheller College of Business lettering.'),
(80,'scheller-college','moderate','Which Georgia Tech college occupies this street-corner building?','Older brick-and-glass Tech Square business building, distinct from the new towers.'),
(39,'george','hard','What is the name of the taller dark-glass tower in the background?','Tall dark-glass George Tower behind the shorter finned tower.'),
(40,'scheller-tower','moderate','Name the shorter white-finned tower in the foreground.','Scheller Tower has prominent white vertical fins; taller George Tower is behind.'),
(41,'scheller-tower','easy','Which tower is named by the large lettering on this facade?','SCHELLER lettering on the white-finned tower.'),
(42,'scheller-tower','moderate','Identify the angular white-finned tower shown from the street.','Stacked projecting volumes and vertical white fins.'),
(44,'george','easy','Which tower is identified by the lettering in this photograph?','GEORGE lettering on the facade.'),
(46,'unknown-tower','hard','Can this close-up of a GT logo alone establish the exact building name?','Only a generic GT logo, dark glass and tree foliage are shown.'),
(47,'tech-tower','easy','Which landmark has these large TECH letters below its roof?','TECH letters on red brick below a pointed roof.'),
(48,'tech-tower','moderate','Which red-brick campus landmark is visible through the trees?','Historic brick tower, arched windows and TECH letters.'),
(49,'tech-tower','hard','Which landmark is suggested by this blurred winter view? Give a tentative identification if needed.','Blurred pointed tower, red brick and pale trim.'),
(50,'tech-tower','easy','Name this landmark with illuminated TECH letters.','TECH letters below the steep tower roof.'),
(52,'tech-tower','moderate','Identify the historic red-brick tower in the foreground, rather than the skyline behind it.','TECH lettering and pointed roof on the foreground brick tower.'),
(54,'tech-tower','hard','Name the historic pointed-roof landmark in front of the large modern high-rise.','Brick tower and TECH lettering below the modern high-rise.'),
(55,'tech-tower','moderate','Which campus landmark is framed by the autumn branches?','TECH lettering and pointed slate roof.'),
(56,'tech-tower','hard','Which landmark is represented by this isolated pointed rooftop?','Pyramidal slate roof and gold tip; most of the building is cropped out.'),
(57,'tech-tower','hard','Identify the small historic tower framed by the dark foreground.','Small but readable TECH letters on a pointed brick tower.'),
(63,'tech-tower','moderate','Which landmark is shown above the flowering branches?','Brick tower, arched windows and steep roof.'),
(64,'tech-tower','easy','What is the name of the red-brick tower at the top of this landscaped slope?','Tech Tower silhouette and TECH letters.'),
(95,'campanile','moderate','Name the slender freestanding landmark at the end of this path.','Thin metal spire in the plaza beyond the trees.'),
(100,'campanile','easy','Which campus landmark rises above the fountain jets?','Metal campanile surrounded by fountain jets.'),
]
NAMES = {
'brittain':('Brittain Dining Hall',['Marion L. Brittain Dining Hall','Brittain']),
'clough':('G. Wayne Clough Undergraduate Learning Commons',['Clough Commons','Clough','CULC']),
'kendeda':('Kendeda Building for Innovative Sustainable Design',['The Kendeda Building','Kendeda']),
'campanile':('Kessler Campanile',['Kessler Campanile Fountain']),
'klaus':('Klaus Advanced Computing Building',['Klaus','KACB','Christopher W. Klaus Advanced Computing Building']),
'mccamish':('McCamish Pavilion',['Hank McCamish Pavilion']),
'carnegie':('Carnegie Building',['Andrew Carnegie Building','Carnegie Library']),
'scheller-college':('Scheller College of Business building',['Scheller College of Business','Georgia Tech College of Management building']),
'scheller-tower':('Scheller Tower',[]),
'george':('George Tower',['Bill and Penny George Tower']),
'unknown-tower':('Exact building cannot be established from this close-up alone.',[]),
 'tech-tower':('Tech Tower',['Lettie Pate Whitehead Evans Administration Building','Evans Administration Building']),
}
# Short paraphrased source cards; these are deliberately supplied to the model
# for evidence-grounded retrieval, not presented as image-visible facts.
SOURCES = {
 'tech': {'url':'https://facilities.gatech.edu/sites/default/files/2023-05/histpresplan.pdf','text':'Tech Tower is the Lettie Pate Whitehead Evans Administration Building. Completed in 1888, it was designed by Bruce and Morgan in the Romanesque Revival style. Its historical roles included teaching and administration.'},
 'clough': {'url':'https://ce.gatech.edu/node/7514','text':'The G. Wayne Clough Undergraduate Learning Commons opened for the fall 2011 semester. It honors former Georgia Tech president G. Wayne Clough and adjoins Price Gilbert Library. It supports undergraduate learning.'},
 'kendeda': {'url':'https://livingbuilding.gatech.edu/about','text':'The Kendeda Building for Innovative Sustainable Design reached substantial completion and soft launch in fall 2019. Classes began in January 2020; Living Building Challenge certification followed in March 2021. Lord Aeck Sargent and The Miller Hull Partnership led architecture; Skanska USA built it.'},
 'campanile': {'url':'https://arts.gatech.edu/public-art/kessler-campanile','text':'The Kessler Campanile is an 80-foot stainless-steel landmark designed by Richard Hill. It was constructed for the 1996 Olympics and stands between the John Lewis Student Center and Tech Green.'},
 'klaus': {'url':'https://sites.cc.gatech.edu/compiler/archive/compiler-november06.html','text':'The Klaus Advanced Computing Building officially opened with a ribbon-cutting on October 26, 2006. A gift from Christopher Klaus made it possible. It supports computing research, including robotics, supercomputing and information security.'},
 'brittain': {'url':'https://www.gatech.edu/news/2026/08/12/brittain-dining-hall-reopening-following-two-year-renovation','text':'Brittain Dining Hall was originally constructed in 1928. Its 2026 reopening followed a two-year renovation. Collegiate Gothic features in its exterior and main dining room were retained while kitchens, service areas and systems were modernized.'},
 'mccamish': {'url':'https://ramblinwreck.com/facilities/mccamish-pavilion','text':'McCamish Pavilion is Georgia Tech\'s basketball arena, named for Hank McCamish. Its first men\'s game was November 9, 2012 against Tulane. It occupies the former Alexander Memorial Coliseum site at 10th and Fowler Streets. The renovation included a rectangular seating bowl and upper deck.'},
 'carnegie': {'url':'https://provost.gatech.edu/news/renewal-gets-real-library-next-site','text':'The Carnegie Building was built in 1907 as Georgia Tech\'s first purpose-built library.'},
 'towers': {'url':'https://news.gatech.edu/news/2026/01/07/tech-square-3-reaches-major-milestone','text':'Tech Square 3 is named George Tower | Scheller Tower. Its first three floors were scheduled to open January 12, 2026. The 14-story Scheller Tower houses graduate and executive business programs; the 18-story George Tower houses the H. Milton Stewart School of Industrial and Systems Engineering.'},
}
# 30 building-record tasks, including short descriptions and multi-fact queries.
# source image, key, source card, difficulty, question, independently graded points
FACTS = [
(47,'tech-tower','tech','easy','What is the full official name of this landmark, according to the supplied record?',['Lettie Pate Whitehead Evans Administration Building']),
(48,'tech-tower','tech','easy','According to the record, in which year was this building completed?',['1888']),
(55,'tech-tower','tech','moderate','Who designed the pictured landmark, and what architectural style does the record assign it?',['Bruce and Morgan','Romanesque Revival']),
(63,'tech-tower','tech','hard','Tell me about this building in two sentences: include its official name, completion year, designers and style from the record.',['Lettie Pate Whitehead Evans Administration Building','1888','Bruce and Morgan','Romanesque Revival']),
(2,'clough','clough','easy','In which year did the building open, according to the supplied record?',['2011']),
(4,'clough','clough','easy','Whom does this building honor, according to the record?',['G. Wayne Clough','Former Georgia Tech president']),
(3,'clough','clough','moderate','Tell me about this building: give its name, opening year and academic purpose using the record.',['Clough Undergraduate Learning Commons','2011','Undergraduate learning']),
(12,'clough','clough','moderate','Which library does this building adjoin, and in which semester did it open?',['Price Gilbert Library','Fall 2011']),
(13,'kendeda','kendeda','easy','What year was this building substantially completed and softly launched?',['2019']),
(18,'kendeda','kendeda','moderate','When did classes begin here, and when did Living Building Challenge certification follow?',['January 2020','March 2021']),
(17,'kendeda','kendeda','moderate','Name the two architecture firms and the builder listed for this building.',['Lord Aeck Sargent','The Miller Hull Partnership','Skanska USA']),
(14,'kendeda','kendeda','hard','Tell me about this building using the record: distinguish its soft launch, first classes and certification dates.',['Kendeda Building for Innovative Sustainable Design','Fall 2019 soft launch','January 2020 first classes','March 2021 certification']),
(15,'kendeda','kendeda','hard','A caption says this building opened in 2021. Is that accurate? Separate completion, classroom occupancy and certification.',['2019 substantial completion/soft launch','2020 classes','2021 certification, not original opening']),
(20,'campanile','campanile','easy','Who designed this landmark, according to the record?',['Richard Hill']),
(36,'campanile','campanile','easy','For which event and year was this landmark constructed?',['1996 Olympics']),
(100,'campanile','campanile','moderate','Tell me about this landmark: name it and give its material and height from the record.',['Kessler Campanile','Stainless steel','80 feet']),
(95,'campanile','campanile','moderate','Which two campus places does the record place this landmark between?',['John Lewis Student Center','Tech Green']),
(35,'klaus','klaus','easy','On what date did this building officially open, according to the record?',['October 26, 2006']),
(34,'klaus','klaus','moderate','Who provided the gift supporting this building, and what is one research field it supports?',['Christopher Klaus','At least one of robotics, supercomputing or information security']),
(21,'klaus','klaus','hard','Summarize the record for this building: include its name, opening date, benefactor and two research fields.',['Klaus Advanced Computing Building','October 26, 2006','Christopher Klaus','Any two of robotics, supercomputing, information security']),
(0,'brittain','brittain','easy','In what year was this dining hall originally constructed?',['1928']),
(1,'brittain','brittain','moderate','Which architectural style was retained, and in what year did the dining hall reopen after renovation?',['Collegiate Gothic','2026']),
(0,'brittain','brittain','hard','Tell me about the dining hall: distinguish its original construction from its recent renovation and name two kinds of modernized facilities.',['Brittain Dining Hall','1928 construction','2026 reopening','Any two: kitchens, service areas, systems']),
(24,'mccamish','mccamish','easy','For which sport is this venue used, and who is its namesake?',['Basketball','Hank McCamish']),
(23,'mccamish','mccamish','moderate','When was the first men\'s game here under the McCamish name, and who was the opponent?',['November 9, 2012','Tulane']),
(25,'mccamish','mccamish','hard','Tell me about this arena using the record: give its name, predecessor and two seating changes.',['McCamish Pavilion','Alexander Memorial Coliseum','Rectangular seating bowl','Upper deck']),
(32,'carnegie','carnegie','easy','In what year was this former library built?',['1907']),
(33,'carnegie','carnegie','moderate','What was this building\'s original role, and what was its construction year?',['First purpose-built Georgia Tech library','1907']),
(41,'scheller-tower','towers','hard','According to the record, which programs use the pictured tower, how many stories does it have, and when were the complex\'s first three floors scheduled to open?',['Graduate and executive business programs','14 stories','January 12, 2026']),
(44,'george','towers','hard','Identify the tower named in this picture and retrieve its school, story count and companion tower from the record.',['George Tower','H. Milton Stewart School of Industrial and Systems Engineering','18 stories','Scheller Tower']),
]
# 12 capture-date tasks with explicitly provided EXIF, not filename inference.
DATES = [0,2,13,16,19,21,22,38,39,47,50,64]
# 8 unanswerable controls: no record/EXIF is supplied to the model.
CONTROLS = [
(0,'brittain','Which exact year was this photograph taken? Use only the visible image.', 'capture_year'),
(13,'kendeda','Can you determine the exact capture year of this photograph from its lighting and architecture alone?', 'capture_year'),
(39,'george','The tower looks new. In what exact year was this photograph taken?', 'capture_year'),
(47,'tech-tower','Which year was this picture taken? Distinguish the photograph date from the building age.', 'capture_year'),
(31,'tech-tower','Give the exact year this historic photograph was taken, using only the picture.', 'capture_year'),
(34,'klaus','What was the exact construction cost of this building? No project records are provided.', 'construction_cost'),
(38,'scheller-college','Who was the photographer, and on what exact date was this photograph taken?', 'photographer_and_date'),
(18,'kendeda','What was the exact number of people enrolled in classes in this building on the day of the photo?', 'enrollment'),
]


def main():
    inventory = json.loads((OUT/'image_inventory.json').read_text())
    (OUT/'images').mkdir(exist_ok=True)
    asset_ids = sorted({r[0] for r in IDENTIFICATIONS+FACTS+CONTROLS} | set(DATES))
    assets = {}
    for i in asset_ids:
        p = ROOT/inventory[i]['path']
        dest = OUT/'images'/f'image_{i:03d}.jpg'
        with Image.open(p) as im:
            clean = ImageOps.exif_transpose(im).convert('RGB')
            clean.thumbnail((2048,2048))
            clean.save(dest,quality=95,exif=b'')
        assets[i] = str(dest.relative_to(ROOT))
        inventory[i]['benchmark_image'] = assets[i]
        inventory[i]['sha256_original'] = hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'image_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    (OUT/'sources.json').write_text(json.dumps(SOURCES,indent=2)+'\n')
    cases=[]
    def add(qtype,idx,loc,diff,q,answer,points,evidence=None,track='visual_identification',answerable=True,avoid=None,source=None):
        number=1+sum(c['query_type']==qtype for c in cases)
        prefix='LI' if qtype=='landmark_identification' else 'FR'
        c=dict(case_id=f'MEI-{prefix}-{number:03d}',dataset=DATASET,location_id=loc,split='test',query_type=qtype,difficulty=diff,answerable=answerable,expected_action='answer' if answerable else 'use_tools',question=q,images=[assets[idx]],reference=dict(answer=answer,required_points=points,unsupported_claims_to_avoid=avoid or ['Invented capture dates, building history or other facts outside the requested answer.']),labels=dict(author='draft_by_codex',reviewed=False,source=source or inventory[idx]['path'],note='AI-drafted after visual inspection; human review pending.',difficulty_rationale=track,source_image=inventory[idx]['path']),hard_negative=not answerable,supplied_evidence=evidence,tags=[track,'human_review_pending'])
        cases.append(c)
        return c
    for idx,loc,diff,q,cue in IDENTIFICATIONS:
        name,aliases=NAMES[loc]
        answerable=loc!='unknown-tower'
        answer=f'{name}. Visible cues: {cue}' if answerable else name+' The GT logo establishes Georgia Tech affiliation, not a unique building identity.'
        points=[f'Identify {name}; accept these aliases: '+', '.join(aliases)+'.'] if answerable else ['State that the exact building is not established by this close-up.','Do not turn the GT logo into a unique building identification.']
        c=add('landmark_identification',idx,loc,diff,q,answer,points,answerable=answerable)
        c['reference']['accepted_aliases']=aliases
        c['labels']['difficulty_rationale']=cue
        if loc in ('george','scheller-tower'): c['reference']['unsupported_claims_to_avoid'] += ['Confusing George Tower, Scheller Tower, and the older Scheller College building.']
        if loc=='tech-tower': c['reference']['unsupported_claims_to_avoid'] += ['Confusing Tech Tower with Kessler Campanile or the modern tower behind it.']
    for idx,loc,src,diff,q,pts in FACTS:
        card=SOURCES[src]
        add('factual_retrieval',idx,loc,diff,q,'; '.join(pts)+'.',pts,evidence='Building record (paraphrased from '+card['url']+'): '+card['text'],track='building_record',source=card['url'])
    for n,idx in enumerate(DATES):
        stamp=inventory[idx]['metadata']['DateTimeOriginal']
        year=stamp[:4];date=stamp[:10].replace(':','-')
        loc=next(r[1] for r in IDENTIFICATIONS if r[0]==idx)
        if n<6:
            q='According to the supplied capture metadata, which year was this picture taken?'
            diff='easy';pts=[year];answer=f'The EXIF DateTimeOriginal field records {year}.'
            evidence=f'EXIF extracted from the original image: DateTimeOriginal={stamp}. This is a camera metadata record, not independently authenticated.'
        else:
            q='Using the supplied record, give the capture date (YYYY-MM-DD). Does the filename prefix establish the capture year?'
            diff='moderate' if n<10 else 'hard';pts=[date,'Use DateTimeOriginal rather than guessing from the filename prefix.'];answer=f'The metadata records {date}. The filename prefix is an asset identifier and does not establish the capture year.'
            evidence=f'Original asset filename: {Path(inventory[idx]["path"]).name}; EXIF DateTimeOriginal={stamp}. This camera metadata is not independently authenticated.'
        add('factual_retrieval',idx,loc,diff,q,answer,pts,evidence=evidence,track='capture_metadata',source=inventory[idx]['path']+'#EXIF:DateTimeOriginal',avoid=['Treating metadata as independently authenticated historical truth.','Using the filename prefix, opening date, or construction year as the photo date.'])
    for idx,loc,q,kind in CONTROLS:
        answer='The exact requested '+kind.replace('_',' ')+' cannot be determined from the supplied photograph alone. A reliable original capture record or relevant external record is needed.'
        add('factual_retrieval',idx,loc,'hard',q,answer,['Explicitly state that the exact requested fact cannot be established from the image alone.','Request suitable metadata or records instead of guessing.'],track='image_only_unanswerable',answerable=False,avoid=['Supplying any exact requested date, number, cost or photographer as established fact.','Confusing a building opening date, renovation date or filename with a photograph capture date.'])
    assert len(cases)==100
    assert Counter(c['query_type'] for c in cases)=={'landmark_identification':50,'factual_retrieval':50}
    errors=validate([Case.from_dict(c) for c in cases])
    assert not errors,errors
    payload=dict(schema_version='2.0',dataset=DATASET,description='50 landmark identification and 50 evidence-grounded factual retrieval cases; AI-drafted, human review pending.',cases=cases)
    (ROOT/'datasets/mapevolve_images.json').write_text(json.dumps(payload,indent=2)+'\n')
    with (OUT/'answer_key.csv').open('w',newline='') as f:
        fields=['case_id','query_type','difficulty','track','image','original_image','question','supplied_evidence','reference_answer','required_points','unsupported_claims_to_avoid','answerable','expected_action','source','reviewed']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for c in cases:
            w.writerow(dict(case_id=c['case_id'],query_type=c['query_type'],difficulty=c['difficulty'],track=c['tags'][0],image=c['images'][0],original_image=c['labels']['source_image'],question=c['question'],supplied_evidence=c['supplied_evidence'] or '',reference_answer=c['reference']['answer'],required_points=' | '.join(c['reference']['required_points']),unsupported_claims_to_avoid=' | '.join(c['reference']['unsupported_claims_to_avoid']),answerable=c['answerable'],expected_action=c['expected_action'],source=c['labels']['source'],reviewed=False))
    lines=['# MapEvolve image benchmark: queries and answer key','','AI-drafted labels; human review pending. See README.md for scoring and limitations.','']
    for c in cases:
        rel=c['images'][0].replace('data/mapevolve_queries/','')
        lines += [f'## {c["case_id"]} · {c["query_type"]} · {c["difficulty"]}', '',f'![{c["case_id"]}]({rel})','',f'**Question:** {c["question"]}','',f'**Supplied evidence:** {c["supplied_evidence"] or "None (image only)"}','',f'**Answer key:** {c["reference"]["answer"]}','','**Required points:**']+['- '+p for p in c['reference']['required_points']]+['',f'**Source:** {c["labels"]["source"]}', '']
    (OUT/'queries_and_answers.md').write_text('\n'.join(lines))
    print('Validated',len(cases),'cases;',len(assets),'neutral images')
    for typ in ('landmark_identification','factual_retrieval'):
        print(typ,dict(Counter(c['difficulty'] for c in cases if c['query_type']==typ)))
    print('Factual tracks:',dict(Counter(c['tags'][0] for c in cases if c['query_type']=='factual_retrieval')))

if __name__=='__main__': main()

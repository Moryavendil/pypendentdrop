thingversion=$(hatch version)

git tag $thingversion

git push

git push origin tag $thingversion
